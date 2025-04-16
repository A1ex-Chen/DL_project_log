@torch.no_grad()
def __call__(self, batch_size: int=1, num_inference_steps: int=100,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]]=None,
    audio_length_in_s: Optional[float]=None, return_dict: bool=True) ->Union[
    AudioPipelineOutput, Tuple]:
    """
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of audio samples to generate.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher-quality audio sample at
                the expense of slower inference.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            audio_length_in_s (`float`, *optional*, defaults to `self.unet.config.sample_size/self.unet.config.sample_rate`):
                The length of the generated audio sample in seconds.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.AudioPipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        from diffusers import DiffusionPipeline
        from scipy.io.wavfile import write

        model_id = "harmonai/maestro-150k"
        pipe = DiffusionPipeline.from_pretrained(model_id)
        pipe = pipe.to("cuda")

        audios = pipe(audio_length_in_s=4.0).audios

        # To save locally
        for i, audio in enumerate(audios):
            write(f"maestro_test_{i}.wav", pipe.unet.sample_rate, audio.transpose())

        # To dislay in google colab
        import IPython.display as ipd

        for audio in audios:
            display(ipd.Audio(audio, rate=pipe.unet.sample_rate))
        ```

        Returns:
            [`~pipelines.AudioPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.AudioPipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated audio.
        """
    if audio_length_in_s is None:
        audio_length_in_s = (self.unet.config.sample_size / self.unet.
            config.sample_rate)
    sample_size = audio_length_in_s * self.unet.config.sample_rate
    down_scale_factor = 2 ** len(self.unet.up_blocks)
    if sample_size < 3 * down_scale_factor:
        raise ValueError(
            f"{audio_length_in_s} is too small. Make sure it's bigger or equal to {3 * down_scale_factor / self.unet.config.sample_rate}."
            )
    original_sample_size = int(sample_size)
    if sample_size % down_scale_factor != 0:
        sample_size = (audio_length_in_s * self.unet.config.sample_rate //
            down_scale_factor + 1) * down_scale_factor
        logger.info(
            f'{audio_length_in_s} is increased to {sample_size / self.unet.config.sample_rate} so that it can be handled by the model. It will be cut to {original_sample_size / self.unet.config.sample_rate} after the denoising process.'
            )
    sample_size = int(sample_size)
    dtype = next(self.unet.parameters()).dtype
    shape = batch_size, self.unet.config.in_channels, sample_size
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.'
            )
    audio = randn_tensor(shape, generator=generator, device=self.
        _execution_device, dtype=dtype)
    self.scheduler.set_timesteps(num_inference_steps, device=audio.device)
    self.scheduler.timesteps = self.scheduler.timesteps.to(dtype)
    for t in self.progress_bar(self.scheduler.timesteps):
        model_output = self.unet(audio, t).sample
        audio = self.scheduler.step(model_output, t, audio).prev_sample
    audio = audio.clamp(-1, 1).float().cpu().numpy()
    audio = audio[:, :, :original_sample_size]
    if not return_dict:
        return audio,
    return AudioPipelineOutput(audios=audio)
