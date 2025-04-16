@torch.no_grad()
def __call__(self, audio, sampling_rate=16000, height: int=512, width: int=
    512, num_inference_steps: int=50, guidance_scale: float=7.5,
    negative_prompt: Optional[Union[str, List[str]]]=None,
    num_images_per_prompt: Optional[int]=1, eta: float=0.0, generator:
    Optional[torch.Generator]=None, latents: Optional[torch.Tensor]=None,
    output_type: Optional[str]='pil', return_dict: bool=True, callback:
    Optional[Callable[[int, int, torch.Tensor], None]]=None, callback_steps:
    int=1, **kwargs):
    inputs = self.speech_processor.feature_extractor(audio, return_tensors=
        'pt', sampling_rate=sampling_rate).input_features.to(self.device)
    predicted_ids = self.speech_model.generate(inputs, max_length=480000)
    prompt = self.speech_processor.tokenizer.batch_decode(predicted_ids,
        skip_special_tokens=True, normalize=True)[0]
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
    if callback_steps is None or callback_steps is not None and (not
        isinstance(callback_steps, int) or callback_steps <= 0):
        raise ValueError(
            f'`callback_steps` has to be a positive integer but is {callback_steps} of type {type(callback_steps)}.'
            )
    text_inputs = self.tokenizer(prompt, padding='max_length', max_length=
        self.tokenizer.model_max_length, return_tensors='pt')
    text_input_ids = text_inputs.input_ids
    if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
        removed_text = self.tokenizer.batch_decode(text_input_ids[:, self.
            tokenizer.model_max_length:])
        logger.warning(
            f'The following part of your input was truncated because CLIP can only handle sequences up to {self.tokenizer.model_max_length} tokens: {removed_text}'
            )
        text_input_ids = text_input_ids[:, :self.tokenizer.model_max_length]
    text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]
    bs_embed, seq_len, _ = text_embeddings.shape
    text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
    text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt,
        seq_len, -1)
    do_classifier_free_guidance = guidance_scale > 1.0
    if do_classifier_free_guidance:
        uncond_tokens: List[str]
        if negative_prompt is None:
            uncond_tokens = [''] * batch_size
        elif type(prompt) is not type(negative_prompt):
            raise TypeError(
                f'`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} != {type(prompt)}.'
                )
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f'`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`: {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches the batch size of `prompt`.'
                )
        else:
            uncond_tokens = negative_prompt
        max_length = text_input_ids.shape[-1]
        uncond_input = self.tokenizer(uncond_tokens, padding='max_length',
            max_length=max_length, truncation=True, return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(
            self.device))[0]
        seq_len = uncond_embeddings.shape[1]
        uncond_embeddings = uncond_embeddings.repeat(1,
            num_images_per_prompt, 1)
        uncond_embeddings = uncond_embeddings.view(batch_size *
            num_images_per_prompt, seq_len, -1)
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
        if callback is not None and i % callback_steps == 0:
            step_idx = i // getattr(self.scheduler, 'order', 1)
            callback(step_idx, t, latents)
    latents = 1 / 0.18215 * latents
    image = self.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    if output_type == 'pil':
        image = self.numpy_to_pil(image)
    if not return_dict:
        return image
    return StableDiffusionPipelineOutput(images=image,
        nsfw_content_detected=None)
