@torch.no_grad()
def __call__(self, alpha: float=1.2, height: Optional[int]=512, width:
    Optional[int]=512, num_inference_steps: Optional[int]=50, generator:
    Optional[torch.Generator]=None, output_type: Optional[str]='pil',
    return_dict: bool=True, guidance_scale: float=7.5, eta: float=0.0):
    """
        Function invoked when calling the pipeline for generation.
        Args:
            alpha (`float`, *optional*, defaults to 1.2):
                The interpolation factor between the original and optimized text embeddings. A value closer to 0
                will resemble the original input image.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `nd.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(
            f'`height` and `width` have to be divisible by 8 but are {height} and {width}.'
            )
    if self.text_embeddings is None:
        raise ValueError(
            'Please run the pipe.train() before trying to generate an image.')
    if self.text_embeddings_orig is None:
        raise ValueError(
            'Please run the pipe.train() before trying to generate an image.')
    text_embeddings = alpha * self.text_embeddings_orig + (1 - alpha
        ) * self.text_embeddings
    do_classifier_free_guidance = guidance_scale > 1.0
    if do_classifier_free_guidance:
        uncond_tokens = ['']
        max_length = self.tokenizer.model_max_length
        uncond_input = self.tokenizer(uncond_tokens, padding='max_length',
            max_length=max_length, truncation=True, return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(
            self.device))[0]
        seq_len = uncond_embeddings.shape[1]
        uncond_embeddings = uncond_embeddings.view(1, seq_len, -1)
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    latents_shape = 1, self.unet.config.in_channels, height // 8, width // 8
    latents_dtype = text_embeddings.dtype
    if self.device.type == 'mps':
        latents = torch.randn(latents_shape, generator=generator, device=
            'cpu', dtype=latents_dtype).to(self.device)
    else:
        latents = torch.randn(latents_shape, generator=generator, device=
            self.device, dtype=latents_dtype)
    self.scheduler.set_timesteps(num_inference_steps)
    timesteps_tensor = self.scheduler.timesteps.to(self.device)
    latents = latents * self.scheduler.init_noise_sigma
    accepts_eta = 'eta' in set(inspect.signature(self.scheduler.step).
        parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs['eta'] = eta
    for i, t in enumerate(self.progress_bar(timesteps_tensor)):
        latent_model_input = torch.cat([latents] * 2
            ) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(
            latent_model_input, t)
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states
            =text_embeddings).sample
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text
                 - noise_pred_uncond)
        latents = self.scheduler.step(noise_pred, t, latents, **
            extra_step_kwargs).prev_sample
    latents = 1 / 0.18215 * latents
    image = self.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    if self.safety_checker is not None:
        safety_checker_input = self.feature_extractor(self.numpy_to_pil(
            image), return_tensors='pt').to(self.device)
        image, has_nsfw_concept = self.safety_checker(images=image,
            clip_input=safety_checker_input.pixel_values.to(text_embeddings
            .dtype))
    else:
        has_nsfw_concept = None
    if output_type == 'pil':
        image = self.numpy_to_pil(image)
    if not return_dict:
        return image, has_nsfw_concept
    return StableDiffusionPipelineOutput(images=image,
        nsfw_content_detected=has_nsfw_concept)
