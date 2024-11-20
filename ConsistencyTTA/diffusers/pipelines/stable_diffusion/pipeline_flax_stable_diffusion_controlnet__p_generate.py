@partial(jax.pmap, in_axes=(None, 0, 0, 0, 0, None, 0, 0, 0, 0),
    static_broadcasted_argnums=(0, 5))
def _p_generate(pipe, prompt_ids, image, params, prng_seed,
    num_inference_steps, guidance_scale, latents, neg_prompt_ids,
    controlnet_conditioning_scale):
    return pipe._generate(prompt_ids, image, params, prng_seed,
        num_inference_steps, guidance_scale, latents, neg_prompt_ids,
        controlnet_conditioning_scale)
