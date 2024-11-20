def __call__(self, prompt: Union[str, List[str]], image: Union[np.ndarray,
    PIL.Image.Image]=None, mask_image: Union[np.ndarray, PIL.Image.Image]=
    None, strength: float=0.8, num_inference_steps: Optional[int]=50,
    guidance_scale: Optional[float]=7.5, negative_prompt: Optional[Union[
    str, List[str]]]=None, num_images_per_prompt: Optional[int]=1, eta:
    Optional[float]=0.0, generator: Optional[np.random.RandomState]=None,
    prompt_embeds: Optional[np.ndarray]=None, negative_prompt_embeds:
    Optional[np.ndarray]=None, output_type: Optional[str]='pil',
    return_dict: bool=True, callback: Optional[Callable[[int, int, np.
    ndarray], None]]=None, callback_steps: int=1):
    """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            image (`nd.ndarray` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process. This is the image whose masked region will be inpainted.
            mask_image (`nd.ndarray` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, to mask `image`. White pixels in the mask will be
                replaced by noise and therefore repainted, while black pixels will be preserved. If `mask_image` is a
                PIL image, it will be converted to a single channel (luminance) before use. If it's a tensor, it should
                contain one color channel (L) instead of 3, so the expected shape would be `(B, H, W, 1)`.uu
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter will be modulated by `strength`.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (?) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`np.random.RandomState`, *optional*):
                A np.random.RandomState to make generation deterministic.
            prompt_embeds (`np.ndarray`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`np.ndarray`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: np.ndarray)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
    self.check_inputs(prompt, callback_steps, negative_prompt,
        prompt_embeds, negative_prompt_embeds)
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    if strength < 0 or strength > 1:
        raise ValueError(
            f'The value of strength should in [0.0, 1.0] but is {strength}')
    if generator is None:
        generator = np.random
    self.scheduler.set_timesteps(num_inference_steps)
    if isinstance(image, PIL.Image.Image):
        image = preprocess(image)
    do_classifier_free_guidance = guidance_scale > 1.0
    prompt_embeds = self._encode_prompt(prompt, num_images_per_prompt,
        do_classifier_free_guidance, negative_prompt, prompt_embeds=
        prompt_embeds, negative_prompt_embeds=negative_prompt_embeds)
    latents_dtype = prompt_embeds.dtype
    image = image.astype(latents_dtype)
    init_latents = self.vae_encoder(sample=image)[0]
    init_latents = 0.18215 * init_latents
    init_latents = np.concatenate([init_latents] * num_images_per_prompt,
        axis=0)
    init_latents_orig = init_latents
    if not isinstance(mask_image, np.ndarray):
        mask_image = preprocess_mask(mask_image, 8)
    mask_image = mask_image.astype(latents_dtype)
    mask = np.concatenate([mask_image] * num_images_per_prompt, axis=0)
    if not mask.shape == init_latents.shape:
        raise ValueError('The mask and image should be the same size!')
    offset = self.scheduler.config.get('steps_offset', 0)
    init_timestep = int(num_inference_steps * strength) + offset
    init_timestep = min(init_timestep, num_inference_steps)
    timesteps = self.scheduler.timesteps.numpy()[-init_timestep]
    timesteps = np.array([timesteps] * batch_size * num_images_per_prompt)
    noise = generator.randn(*init_latents.shape).astype(latents_dtype)
    init_latents = self.scheduler.add_noise(torch.from_numpy(init_latents),
        torch.from_numpy(noise), torch.from_numpy(timesteps))
    init_latents = init_latents.numpy()
    accepts_eta = 'eta' in set(inspect.signature(self.scheduler.step).
        parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs['eta'] = eta
    latents = init_latents
    t_start = max(num_inference_steps - init_timestep + offset, 0)
    timesteps = self.scheduler.timesteps[t_start:].numpy()
    timestep_dtype = next((input.type for input in self.unet.model.
        get_inputs() if input.name == 'timestep'), 'tensor(float)')
    timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]
    for i, t in enumerate(self.progress_bar(timesteps)):
        latent_model_input = np.concatenate([latents] * 2
            ) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(
            latent_model_input, t)
        timestep = np.array([t], dtype=timestep_dtype)
        noise_pred = self.unet(sample=latent_model_input, timestep=timestep,
            encoder_hidden_states=prompt_embeds)[0]
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text
                 - noise_pred_uncond)
        latents = self.scheduler.step(torch.from_numpy(noise_pred), t,
            torch.from_numpy(latents), **extra_step_kwargs).prev_sample
        latents = latents.numpy()
        init_latents_proper = self.scheduler.add_noise(torch.from_numpy(
            init_latents_orig), torch.from_numpy(noise), torch.from_numpy(
            np.array([t])))
        init_latents_proper = init_latents_proper.numpy()
        latents = init_latents_proper * mask + latents * (1 - mask)
        if callback is not None and i % callback_steps == 0:
            step_idx = i // getattr(self.scheduler, 'order', 1)
            callback(step_idx, t, latents)
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
