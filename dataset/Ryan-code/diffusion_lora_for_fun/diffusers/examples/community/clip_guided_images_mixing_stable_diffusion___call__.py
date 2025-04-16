@torch.no_grad()
def __call__(self, style_image: Union[torch.Tensor, PIL.Image.Image],
    content_image: Union[torch.Tensor, PIL.Image.Image], style_prompt:
    Optional[str]=None, content_prompt: Optional[str]=None, height:
    Optional[int]=512, width: Optional[int]=512, noise_strength: float=0.6,
    num_inference_steps: Optional[int]=50, guidance_scale: Optional[float]=
    7.5, batch_size: Optional[int]=1, eta: float=0.0, clip_guidance_scale:
    Optional[float]=100, generator: Optional[torch.Generator]=None,
    output_type: Optional[str]='pil', return_dict: bool=True,
    slerp_latent_style_strength: float=0.8, slerp_prompt_style_strength:
    float=0.1, slerp_clip_image_style_strength: float=0.1):
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f'You have passed {batch_size} batch_size, but only {len(generator)} generators.'
            )
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(
            f'`height` and `width` have to be divisible by 8 but are {height} and {width}.'
            )
    if isinstance(generator, torch.Generator) and batch_size > 1:
        generator = [generator] + [None] * (batch_size - 1)
    coca_is_none = [('model', self.coca_model is None), ('tokenizer', self.
        coca_tokenizer is None), ('transform', self.coca_transform is None)]
    coca_is_none = [x[0] for x in coca_is_none if x[1]]
    coca_is_none_str = ', '.join(coca_is_none)
    if content_prompt is None:
        if len(coca_is_none):
            raise ValueError(
                f'Content prompt is None and CoCa [{coca_is_none_str}] is None.Set prompt or pass Coca [{coca_is_none_str}] to DiffusionPipeline.'
                )
        content_prompt = self.get_image_description(content_image)
    if style_prompt is None:
        if len(coca_is_none):
            raise ValueError(
                f'Style prompt is None and CoCa [{coca_is_none_str}] is None. Set prompt or pass Coca [{coca_is_none_str}] to DiffusionPipeline.'
                )
        style_prompt = self.get_image_description(style_image)
    content_text_input = self.tokenizer(content_prompt, padding=
        'max_length', max_length=self.tokenizer.model_max_length,
        truncation=True, return_tensors='pt')
    content_text_embeddings = self.text_encoder(content_text_input.
        input_ids.to(self.device))[0]
    style_text_input = self.tokenizer(style_prompt, padding='max_length',
        max_length=self.tokenizer.model_max_length, truncation=True,
        return_tensors='pt')
    style_text_embeddings = self.text_encoder(style_text_input.input_ids.to
        (self.device))[0]
    text_embeddings = slerp(slerp_prompt_style_strength,
        content_text_embeddings, style_text_embeddings)
    text_embeddings = text_embeddings.repeat_interleave(batch_size, dim=0)
    accepts_offset = 'offset' in set(inspect.signature(self.scheduler.
        set_timesteps).parameters.keys())
    extra_set_kwargs = {}
    if accepts_offset:
        extra_set_kwargs['offset'] = 1
    self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
    self.scheduler.timesteps.to(self.device)
    timesteps, num_inference_steps = self.get_timesteps(num_inference_steps,
        noise_strength, self.device)
    latent_timestep = timesteps[:1].repeat(batch_size)
    preprocessed_content_image = preprocess(content_image, width, height)
    content_latents = self.prepare_latents(preprocessed_content_image,
        latent_timestep, batch_size, text_embeddings.dtype, self.device,
        generator)
    preprocessed_style_image = preprocess(style_image, width, height)
    style_latents = self.prepare_latents(preprocessed_style_image,
        latent_timestep, batch_size, text_embeddings.dtype, self.device,
        generator)
    latents = slerp(slerp_latent_style_strength, content_latents, style_latents
        )
    if clip_guidance_scale > 0:
        content_clip_image_embedding = self.get_clip_image_embeddings(
            content_image, batch_size)
        style_clip_image_embedding = self.get_clip_image_embeddings(style_image
            , batch_size)
        clip_image_embeddings = slerp(slerp_clip_image_style_strength,
            content_clip_image_embedding, style_clip_image_embedding)
    do_classifier_free_guidance = guidance_scale > 1.0
    if do_classifier_free_guidance:
        max_length = content_text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer([''], padding='max_length',
            max_length=max_length, return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(
            self.device))[0]
        uncond_embeddings = uncond_embeddings.repeat_interleave(batch_size,
            dim=0)
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    latents_shape = (batch_size, self.unet.config.in_channels, height // 8,
        width // 8)
    latents_dtype = text_embeddings.dtype
    if latents is None:
        if self.device.type == 'mps':
            latents = torch.randn(latents_shape, generator=generator,
                device='cpu', dtype=latents_dtype).to(self.device)
        else:
            latents = torch.randn(latents_shape, generator=generator,
                device=self.device, dtype=latents_dtype)
    else:
        if latents.shape != latents_shape:
            raise ValueError(
                f'Unexpected latents shape, got {latents.shape}, expected {latents_shape}'
                )
        latents = latents.to(self.device)
    latents = latents * self.scheduler.init_noise_sigma
    accepts_eta = 'eta' in set(inspect.signature(self.scheduler.step).
        parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs['eta'] = eta
    accepts_generator = 'generator' in set(inspect.signature(self.scheduler
        .step).parameters.keys())
    if accepts_generator:
        extra_step_kwargs['generator'] = generator
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2
                ) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)
            noise_pred = self.unet(latent_model_input, t,
                encoder_hidden_states=text_embeddings).sample
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
            if clip_guidance_scale > 0:
                text_embeddings_for_guidance = text_embeddings.chunk(2)[1
                    ] if do_classifier_free_guidance else text_embeddings
                noise_pred, latents = self.cond_fn(latents, t, i,
                    text_embeddings_for_guidance, noise_pred,
                    clip_image_embeddings, clip_guidance_scale)
            latents = self.scheduler.step(noise_pred, t, latents, **
                extra_step_kwargs).prev_sample
            progress_bar.update()
    latents = 1 / 0.18215 * latents
    image = self.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    if output_type == 'pil':
        image = self.numpy_to_pil(image)
    if not return_dict:
        return image, None
    return StableDiffusionPipelineOutput(images=image,
        nsfw_content_detected=None)
