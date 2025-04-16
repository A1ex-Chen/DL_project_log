@torch.no_grad()
def __call__(self, batch_size: int=1, audio_file: str=None, raw_audio: np.
    ndarray=None, slice: int=0, start_step: int=0, steps: int=None,
    generator: torch.Generator=None, mask_start_secs: float=0,
    mask_end_secs: float=0, step_generator: torch.Generator=None, eta:
    float=0, noise: torch.Tensor=None, encoding: torch.Tensor=None,
    return_dict=True) ->Union[Union[AudioPipelineOutput,
    ImagePipelineOutput], Tuple[List[Image.Image], Tuple[int, List[np.
    ndarray]]]]:
    """Generate random mel spectrogram from audio input and convert to audio.

        Args:
            batch_size (`int`): number of samples to generate
            audio_file (`str`): must be a file on disk due to Librosa limitation or
            raw_audio (`np.ndarray`): audio as numpy array
            slice (`int`): slice number of audio to convert
            start_step (int): step to start from
            steps (`int`): number of de-noising steps (defaults to 50 for DDIM, 1000 for DDPM)
            generator (`torch.Generator`): random number generator or None
            mask_start_secs (`float`): number of seconds of audio to mask (not generate) at start
            mask_end_secs (`float`): number of seconds of audio to mask (not generate) at end
            step_generator (`torch.Generator`): random number generator used to de-noise or None
            eta (`float`): parameter between 0 and 1 used with DDIM scheduler
            noise (`torch.Tensor`): noise tensor of shape (batch_size, 1, height, width) or None
            encoding (`torch.Tensor`): for UNet2DConditionModel shape (batch_size, seq_length, cross_attention_dim)
            return_dict (`bool`): if True return AudioPipelineOutput, ImagePipelineOutput else Tuple

        Returns:
            `List[PIL Image]`: mel spectrograms (`float`, `List[np.ndarray]`): sample rate and raw audios
        """
    steps = steps or self.get_default_steps()
    self.scheduler.set_timesteps(steps)
    step_generator = step_generator or generator
    if type(self.unet.sample_size) == int:
        self.unet.sample_size = self.unet.sample_size, self.unet.sample_size
    input_dims = self.get_input_dims()
    self.mel.set_resolution(x_res=input_dims[1], y_res=input_dims[0])
    if noise is None:
        noise = randn_tensor((batch_size, self.unet.in_channels, self.unet.
            sample_size[0], self.unet.sample_size[1]), generator=generator,
            device=self.device)
    images = noise
    mask = None
    if audio_file is not None or raw_audio is not None:
        self.mel.load_audio(audio_file, raw_audio)
        input_image = self.mel.audio_slice_to_image(slice)
        input_image = np.frombuffer(input_image.tobytes(), dtype='uint8'
            ).reshape((input_image.height, input_image.width))
        input_image = input_image / 255 * 2 - 1
        input_images = torch.tensor(input_image[np.newaxis, :, :], dtype=
            torch.float).to(self.device)
        if self.vqvae is not None:
            input_images = self.vqvae.encode(torch.unsqueeze(input_images, 0)
                ).latent_dist.sample(generator=generator)[0]
            input_images = self.vqvae.config.scaling_factor * input_images
        if start_step > 0:
            images[0, 0] = self.scheduler.add_noise(input_images, noise,
                self.scheduler.timesteps[start_step - 1])
        pixels_per_second = self.unet.sample_size[1
            ] * self.mel.get_sample_rate(
            ) / self.mel.x_res / self.mel.hop_length
        mask_start = int(mask_start_secs * pixels_per_second)
        mask_end = int(mask_end_secs * pixels_per_second)
        mask = self.scheduler.add_noise(input_images, noise, torch.tensor(
            self.scheduler.timesteps[start_step:]))
    for step, t in enumerate(self.progress_bar(self.scheduler.timesteps[
        start_step:])):
        if isinstance(self.unet, UNet2DConditionModel):
            model_output = self.unet(images, t, encoding)['sample']
        else:
            model_output = self.unet(images, t)['sample']
        if isinstance(self.scheduler, DDIMScheduler):
            images = self.scheduler.step(model_output=model_output,
                timestep=t, sample=images, eta=eta, generator=step_generator)[
                'prev_sample']
        else:
            images = self.scheduler.step(model_output=model_output,
                timestep=t, sample=images, generator=step_generator)[
                'prev_sample']
        if mask is not None:
            if mask_start > 0:
                images[:, :, :, :mask_start] = mask[:, step, :, :mask_start]
            if mask_end > 0:
                images[:, :, :, -mask_end:] = mask[:, step, :, -mask_end:]
    if self.vqvae is not None:
        images = 1 / self.vqvae.config.scaling_factor * images
        images = self.vqvae.decode(images)['sample']
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype('uint8')
    images = list((Image.fromarray(_[:, :, 0]) for _ in images) if images.
        shape[3] == 1 else (Image.fromarray(_, mode='RGB').convert('L') for
        _ in images))
    audios = [self.mel.image_to_audio(_) for _ in images]
    if not return_dict:
        return images, (self.mel.get_sample_rate(), audios)
    return BaseOutput(**AudioPipelineOutput(np.array(audios)[:, np.newaxis,
        :]), **ImagePipelineOutput(images))
