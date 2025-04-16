def __call__(self, prompt: Union[str, List[str]], image: Union[torch.
    FloatTensor, PIL.Image.Image, List[PIL.Image.Image]],
    num_inference_steps: int=75, guidance_scale: float=9.0, noise_level:
    int=20, negative_prompt: Optional[Union[str, List[str]]]=None,
    num_images_per_prompt: Optional[int]=1, eta: float=0.0, generator:
    Optional[Union[torch.Generator, List[torch.Generator]]]=None, latents:
    Optional[torch.FloatTensor]=None, output_type: Optional[str]='pil',
    return_dict: bool=True, callback: Optional[Callable[[int, int, torch.
    FloatTensor], None]]=None, callback_steps: Optional[int]=1):
    self.check_inputs(prompt, image, noise_level, callback_steps)
    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0
    text_embeddings = self._encode_prompt(prompt, device,
        num_images_per_prompt, do_classifier_free_guidance, negative_prompt)
    latents_dtype = ORT_TO_PT_TYPE[str(text_embeddings.dtype)]
    image = preprocess(image)
    image = image.cpu()
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps
    noise_level = torch.tensor([noise_level], dtype=torch.long, device=device)
    noise = torch.randn(image.shape, generator=generator, device=device,
        dtype=latents_dtype)
    image = self.low_res_scheduler.add_noise(image, noise, noise_level)
    batch_multiplier = 2 if do_classifier_free_guidance else 1
    image = np.concatenate([image] * batch_multiplier * num_images_per_prompt)
    noise_level = np.concatenate([noise_level] * image.shape[0])
    height, width = image.shape[2:]
    latents = self.prepare_latents(batch_size * num_images_per_prompt,
        NUM_LATENT_CHANNELS, height, width, latents_dtype, device,
        generator, latents)
    num_channels_image = image.shape[1]
    if NUM_LATENT_CHANNELS + num_channels_image != NUM_UNET_INPUT_CHANNELS:
        raise ValueError(
            f'Incorrect configuration settings! The config of `pipeline.unet` expects {NUM_UNET_INPUT_CHANNELS} but received `num_channels_latents`: {NUM_LATENT_CHANNELS} + `num_channels_image`: {num_channels_image}  = {NUM_LATENT_CHANNELS + num_channels_image}. Please verify the config of `pipeline.unet` or your `image` input.'
            )
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    timestep_dtype = next((input.type for input in self.unet.model.
        get_inputs() if input.name == 'timestep'), 'tensor(float)')
    timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]
    num_warmup_steps = len(timesteps
        ) - num_inference_steps * self.scheduler.order
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = np.concatenate([latents] * 2
                ) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)
            latent_model_input = np.concatenate([latent_model_input, image],
                axis=1)
            timestep = np.array([t], dtype=timestep_dtype)
            noise_pred = self.unet(sample=latent_model_input, timestep=
                timestep, encoder_hidden_states=text_embeddings,
                class_labels=noise_level.astype(np.int64))[0]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text
                 - noise_pred_uncond)
            latents = self.scheduler.step(torch.from_numpy(noise_pred), t,
                latents, **extra_step_kwargs).prev_sample
            if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (i + 1
                ) % self.scheduler.order == 0:
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
    image = self.decode_latents(latents.float())
    if output_type == 'pil':
        image = self.numpy_to_pil(image)
    if not return_dict:
        return image,
    return ImagePipelineOutput(images=image)
