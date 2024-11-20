@torch.no_grad()
def __call__(self, prompt: Union[str, List[str]], height: Optional[int]=512,
    width: Optional[int]=512, num_inference_steps: Optional[int]=50,
    guidance_scale: Optional[float]=7.5, num_images_per_prompt: Optional[
    int]=1, eta: float=0.0, clip_guidance_scale: Optional[float]=100,
    clip_prompt: Optional[Union[str, List[str]]]=None, num_cutouts:
    Optional[int]=4, use_cutouts: Optional[bool]=True, generator: Optional[
    torch.Generator]=None, latents: Optional[torch.Tensor]=None,
    output_type: Optional[str]='pil', return_dict: bool=True):
    if isinstance(prompt, str):
        batch_size = 1
    elif isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        raise ValueError(
            f'`prompt` has to be of type `str` or `list` but is {type(prompt)}'
            )
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(
            f'`height` and `width` have to be divisible by 8 but are {height} and {width}.'
            )
    text_input = self.tokenizer(prompt, padding='max_length', max_length=
        self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
    text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0
        ]
    text_embeddings = text_embeddings.repeat_interleave(num_images_per_prompt,
        dim=0)
    if clip_guidance_scale > 0:
        if clip_prompt is not None:
            clip_text_input = self.tokenizer(clip_prompt, padding=
                'max_length', max_length=self.tokenizer.model_max_length,
                truncation=True, return_tensors='pt').input_ids.to(self.device)
        else:
            clip_text_input = text_input.input_ids.to(self.device)
        text_embeddings_clip = self.clip_model.get_text_features(
            clip_text_input)
        text_embeddings_clip = (text_embeddings_clip / text_embeddings_clip
            .norm(p=2, dim=-1, keepdim=True))
        text_embeddings_clip = text_embeddings_clip.repeat_interleave(
            num_images_per_prompt, dim=0)
    do_classifier_free_guidance = guidance_scale > 1.0
    if do_classifier_free_guidance:
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer([''], padding='max_length',
            max_length=max_length, return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(
            self.device))[0]
        uncond_embeddings = uncond_embeddings.repeat_interleave(
            num_images_per_prompt, dim=0)
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    latents_shape = (batch_size * num_images_per_prompt, self.unet.config.
        in_channels, height // 8, width // 8)
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
    accepts_offset = 'offset' in set(inspect.signature(self.scheduler.
        set_timesteps).parameters.keys())
    extra_set_kwargs = {}
    if accepts_offset:
        extra_set_kwargs['offset'] = 1
    self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
    timesteps_tensor = self.scheduler.timesteps.to(self.device)
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
        if clip_guidance_scale > 0:
            text_embeddings_for_guidance = text_embeddings.chunk(2)[1
                ] if do_classifier_free_guidance else text_embeddings
            noise_pred, latents = self.cond_fn(latents, t, i,
                text_embeddings_for_guidance, noise_pred,
                text_embeddings_clip, clip_guidance_scale, num_cutouts,
                use_cutouts)
        latents = self.scheduler.step(noise_pred, t, latents, **
            extra_step_kwargs).prev_sample
    latents = 1 / self.vae.config.scaling_factor * latents
    image = self.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    if output_type == 'pil':
        image = self.numpy_to_pil(image)
    if not return_dict:
        return image, None
    return StableDiffusionPipelineOutput(images=image,
        nsfw_content_detected=None)
