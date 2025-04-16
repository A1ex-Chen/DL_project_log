@partial(jax.pmap, in_axes=(None, 0, 0, 0, None, None, None, 0, 0, 0, None),
    static_broadcasted_argnums=(0, 4, 5, 6, 10))
def _p_generate(pipe, prompt_ids, params, prng_seed, num_inference_steps,
    height, width, guidance_scale, latents, neg_prompt_ids, return_latents):
    return pipe._generate(prompt_ids, params, prng_seed,
        num_inference_steps, height, width, guidance_scale, latents,
        neg_prompt_ids, return_latents)
