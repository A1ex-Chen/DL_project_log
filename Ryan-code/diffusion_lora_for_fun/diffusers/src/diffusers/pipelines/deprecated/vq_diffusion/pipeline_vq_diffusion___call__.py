@torch.no_grad()
def __call__(self, prompt: Union[str, List[str]], num_inference_steps: int=
    100, guidance_scale: float=5.0, truncation_rate: float=1.0,
    num_images_per_prompt: int=1, generator: Optional[Union[torch.Generator,
    List[torch.Generator]]]=None, latents: Optional[torch.Tensor]=None,
    output_type: Optional[str]='pil', return_dict: bool=True, callback:
    Optional[Callable[[int, int, torch.Tensor], None]]=None, callback_steps:
    int=1) ->Union[ImagePipelineOutput, Tuple]:
    """
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide image generation.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            truncation_rate (`float`, *optional*, defaults to 1.0 (equivalent to no truncation)):
                Used to "truncate" the predicted classes for x_0 such that the cumulative probability for a pixel is at
                most `truncation_rate`. The lowest probabilities that would increase the cumulative probability above
                `truncation_rate` are set to zero.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor` of shape (batch), *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Must be valid embedding indices.If not provided, a latents tensor will be generated of
                completely masked latent pixels.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
        """
    if isinstance(prompt, str):
        batch_size = 1
    elif isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        raise ValueError(
            f'`prompt` has to be of type `str` or `list` but is {type(prompt)}'
            )
    batch_size = batch_size * num_images_per_prompt
    do_classifier_free_guidance = guidance_scale > 1.0
    prompt_embeds = self._encode_prompt(prompt, num_images_per_prompt,
        do_classifier_free_guidance)
    if callback_steps is None or callback_steps is not None and (not
        isinstance(callback_steps, int) or callback_steps <= 0):
        raise ValueError(
            f'`callback_steps` has to be a positive integer but is {callback_steps} of type {type(callback_steps)}.'
            )
    latents_shape = batch_size, self.transformer.num_latent_pixels
    if latents is None:
        mask_class = self.transformer.num_vector_embeds - 1
        latents = torch.full(latents_shape, mask_class).to(self.device)
    else:
        if latents.shape != latents_shape:
            raise ValueError(
                f'Unexpected latents shape, got {latents.shape}, expected {latents_shape}'
                )
        if (latents < 0).any() or (latents >= self.transformer.
            num_vector_embeds).any():
            raise ValueError(
                f'Unexpected latents value(s). All latents be valid embedding indices i.e. in the range 0, {self.transformer.num_vector_embeds - 1} (inclusive).'
                )
        latents = latents.to(self.device)
    self.scheduler.set_timesteps(num_inference_steps, device=self.device)
    timesteps_tensor = self.scheduler.timesteps.to(self.device)
    sample = latents
    for i, t in enumerate(self.progress_bar(timesteps_tensor)):
        latent_model_input = torch.cat([sample] * 2
            ) if do_classifier_free_guidance else sample
        model_output = self.transformer(latent_model_input,
            encoder_hidden_states=prompt_embeds, timestep=t).sample
        if do_classifier_free_guidance:
            model_output_uncond, model_output_text = model_output.chunk(2)
            model_output = model_output_uncond + guidance_scale * (
                model_output_text - model_output_uncond)
            model_output -= torch.logsumexp(model_output, dim=1, keepdim=True)
        model_output = self.truncate(model_output, truncation_rate)
        model_output = model_output.clamp(-70)
        sample = self.scheduler.step(model_output, timestep=t, sample=
            sample, generator=generator).prev_sample
        if callback is not None and i % callback_steps == 0:
            callback(i, t, sample)
    embedding_channels = self.vqvae.config.vq_embed_dim
    embeddings_shape = (batch_size, self.transformer.height, self.
        transformer.width, embedding_channels)
    embeddings = self.vqvae.quantize.get_codebook_entry(sample, shape=
        embeddings_shape)
    image = self.vqvae.decode(embeddings, force_not_quantize=True).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    if output_type == 'pil':
        image = self.numpy_to_pil(image)
    if not return_dict:
        return image,
    return ImagePipelineOutput(images=image)
