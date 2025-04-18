@replace_example_docstring(EXAMPLE_DOC_STRING)
def __call__(self, prompt_ids: jnp.ndarray, image: jnp.ndarray, params:
    Union[Dict, FrozenDict], prng_seed: jax.Array, strength: float=0.8,
    num_inference_steps: int=50, height: Optional[int]=None, width:
    Optional[int]=None, guidance_scale: Union[float, jnp.ndarray]=7.5,
    noise: jnp.ndarray=None, neg_prompt_ids: jnp.ndarray=None, return_dict:
    bool=True, jit: bool=False):
    """
        The call function to the pipeline for generation.

        Args:
            prompt_ids (`jnp.ndarray`):
                The prompt or prompts to guide image generation.
            image (`jnp.ndarray`):
                Array representing an image batch to be used as the starting point.
            params (`Dict` or `FrozenDict`):
                Dictionary containing the model parameters/weights.
            prng_seed (`jax.Array` or `jax.Array`):
                Array containing random number generator key.
            strength (`float`, *optional*, defaults to 0.8):
                Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a
                starting point and more noise is added the higher the `strength`. The number of denoising steps depends
                on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising
                process runs for the full number of iterations specified in `num_inference_steps`. A value of 1
                essentially ignores `image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            noise (`jnp.ndarray`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. The array is generated by
                sampling using the supplied random `generator`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput`] instead of
                a plain tuple.
            jit (`bool`, defaults to `False`):
                Whether to run `pmap` versions of the generation and safety scoring functions.

                    <Tip warning={true}>

                    This argument exists because `__call__` is not yet end-to-end pmap-able. It will be removed in a
                    future release.

                    </Tip>

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput`] is
                returned, otherwise a `tuple` is returned where the first element is a list with the generated images
                and the second element is a list of `bool`s indicating whether the corresponding generated image
                contains "not-safe-for-work" (nsfw) content.
        """
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor
    if isinstance(guidance_scale, float):
        guidance_scale = jnp.array([guidance_scale] * prompt_ids.shape[0])
        if len(prompt_ids.shape) > 2:
            guidance_scale = guidance_scale[:, None]
    start_timestep = self.get_timestep_start(num_inference_steps, strength)
    if jit:
        images = _p_generate(self, prompt_ids, image, params, prng_seed,
            start_timestep, num_inference_steps, height, width,
            guidance_scale, noise, neg_prompt_ids)
    else:
        images = self._generate(prompt_ids, image, params, prng_seed,
            start_timestep, num_inference_steps, height, width,
            guidance_scale, noise, neg_prompt_ids)
    if self.safety_checker is not None:
        safety_params = params['safety_checker']
        images_uint8_casted = (images * 255).round().astype('uint8')
        num_devices, batch_size = images.shape[:2]
        images_uint8_casted = np.asarray(images_uint8_casted).reshape(
            num_devices * batch_size, height, width, 3)
        images_uint8_casted, has_nsfw_concept = self._run_safety_checker(
            images_uint8_casted, safety_params, jit)
        images = np.asarray(images)
        if any(has_nsfw_concept):
            for i, is_nsfw in enumerate(has_nsfw_concept):
                if is_nsfw:
                    images[i] = np.asarray(images_uint8_casted[i])
        images = images.reshape(num_devices, batch_size, height, width, 3)
    else:
        images = np.asarray(images)
        has_nsfw_concept = False
    if not return_dict:
        return images, has_nsfw_concept
    return FlaxStableDiffusionPipelineOutput(images=images,
        nsfw_content_detected=has_nsfw_concept)
