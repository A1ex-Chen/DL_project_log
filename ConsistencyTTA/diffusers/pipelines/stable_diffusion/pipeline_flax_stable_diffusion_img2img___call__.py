@replace_example_docstring(EXAMPLE_DOC_STRING)
def __call__(self, prompt_ids: jnp.array, image: jnp.array, params: Union[
    Dict, FrozenDict], prng_seed: jax.random.KeyArray, strength: float=0.8,
    num_inference_steps: int=50, height: Optional[int]=None, width:
    Optional[int]=None, guidance_scale: Union[float, jnp.array]=7.5, noise:
    jnp.array=None, neg_prompt_ids: jnp.array=None, return_dict: bool=True,
    jit: bool=False):
    """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt_ids (`jnp.array`):
                The prompt or prompts to guide the image generation.
            image (`jnp.array`):
                Array representing an image batch, that will be used as the starting point for the process.
            params (`Dict` or `FrozenDict`): Dictionary containing the model parameters/weights
            prng_seed (`jax.random.KeyArray` or `jax.Array`): Array containing random number generator key
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            noise (`jnp.array`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. tensor will ge generated
                by sampling using the supplied random `generator`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput`] instead of
                a plain tuple.
            jit (`bool`, defaults to `False`):
                Whether to run `pmap` versions of the generation and safety scoring functions. NOTE: This argument
                exists because `__call__` is not yet end-to-end pmap-able. It will be removed in a future release.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple. When returning a tuple, the first element is a list with the generated images, and the second
            element is a list of `bool`s denoting whether the corresponding generated image likely represents
            "not-safe-for-work" (nsfw) content, according to the `safety_checker`.
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
