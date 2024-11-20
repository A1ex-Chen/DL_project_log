def __call__(self, prompt_ids: jax.Array, params: Union[Dict, FrozenDict],
    prng_seed: jax.Array, num_inference_steps: int=50, guidance_scale:
    Union[float, jax.Array]=7.5, height: Optional[int]=None, width:
    Optional[int]=None, latents: jnp.array=None, neg_prompt_ids: jnp.array=
    None, return_dict: bool=True, output_type: str=None, jit: bool=False):
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor
    if isinstance(guidance_scale, float) and jit:
        guidance_scale = jnp.array([guidance_scale] * prompt_ids.shape[0])
        guidance_scale = guidance_scale[:, None]
    return_latents = output_type == 'latent'
    if jit:
        images = _p_generate(self, prompt_ids, params, prng_seed,
            num_inference_steps, height, width, guidance_scale, latents,
            neg_prompt_ids, return_latents)
    else:
        images = self._generate(prompt_ids, params, prng_seed,
            num_inference_steps, height, width, guidance_scale, latents,
            neg_prompt_ids, return_latents)
    if not return_dict:
        return images,
    return FlaxStableDiffusionXLPipelineOutput(images=images)
