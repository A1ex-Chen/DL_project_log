def __call__(self, prompt: Union[str, List[str]], height: Optional[int]=512,
    width: Optional[int]=512, num_inference_steps: Optional[int]=50,
    guidance_scale: Optional[float]=7.5, negative_prompt: Optional[Union[
    str, List[str]]]=None, num_images_per_prompt: Optional[int]=1, eta:
    Optional[float]=0.0, generator: Optional[np.random.RandomState]=None,
    latents: Optional[np.ndarray]=None, output_type: Optional[str]='pil',
    return_dict: bool=True, callback: Optional[Callable[[int, int, np.
    ndarray], None]]=None, callback_steps: int=1):
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
    if generator is None:
        generator = np.random
    do_classifier_free_guidance = guidance_scale > 1.0
    prompt_embeds = self._encode_prompt(prompt, num_images_per_prompt,
        do_classifier_free_guidance, negative_prompt)
    latents_dtype = prompt_embeds.dtype
    latents_shape = (batch_size * num_images_per_prompt, 4, height // 8, 
        width // 8)
    if latents is None:
        latents = generator.randn(*latents_shape).astype(latents_dtype)
    elif latents.shape != latents_shape:
        raise ValueError(
            f'Unexpected latents shape, got {latents.shape}, expected {latents_shape}'
            )
    self.scheduler.set_timesteps(num_inference_steps)
    latents = latents * np.float64(self.scheduler.init_noise_sigma)
    accepts_eta = 'eta' in set(inspect.signature(self.scheduler.step).
        parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs['eta'] = eta
    timestep_dtype = next((input.type for input in self.unet.model.
        get_inputs() if input.name == 'timestep'), 'tensor(float)')
    timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]
    for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
        latent_model_input = np.concatenate([latents] * 2
            ) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(torch.
            from_numpy(latent_model_input), t)
        latent_model_input = latent_model_input.cpu().numpy()
        timestep = np.array([t], dtype=timestep_dtype)
        noise_pred = self.unet(sample=latent_model_input, timestep=timestep,
            encoder_hidden_states=prompt_embeds)
        noise_pred = noise_pred[0]
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text
                 - noise_pred_uncond)
        scheduler_output = self.scheduler.step(torch.from_numpy(noise_pred),
            t, torch.from_numpy(latents), **extra_step_kwargs)
        latents = scheduler_output.prev_sample.numpy()
        if callback is not None and i % callback_steps == 0:
            callback(i, t, latents)
    latents = 1 / 0.18215 * latents
    image = np.concatenate([self.vae_decoder(latent_sample=latents[i:i + 1]
        )[0] for i in range(latents.shape[0])])
    image = np.clip(image / 2 + 0.5, 0, 1)
    image = image.transpose((0, 2, 3, 1))
    if self.safety_checker is not None:
        safety_checker_input = self.feature_extractor(self.numpy_to_pil(
            image), return_tensors='np').pixel_values.astype(image.dtype)
        images, has_nsfw_concept = [], []
        for i in range(image.shape[0]):
            image_i, has_nsfw_concept_i = self.safety_checker(clip_input=
                safety_checker_input[i:i + 1], images=image[i:i + 1])
            images.append(image_i)
            has_nsfw_concept.append(has_nsfw_concept_i[0])
        image = np.concatenate(images)
    else:
        has_nsfw_concept = None
    if output_type == 'pil':
        image = self.numpy_to_pil(image)
    if not return_dict:
        return image, has_nsfw_concept
    return StableDiffusionPipelineOutput(images=image,
        nsfw_content_detected=has_nsfw_concept)
