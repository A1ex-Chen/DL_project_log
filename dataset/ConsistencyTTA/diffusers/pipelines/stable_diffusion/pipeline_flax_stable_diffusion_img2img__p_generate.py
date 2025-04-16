@partial(jax.pmap, in_axes=(None, 0, 0, 0, 0, None, None, None, None, 0, 0,
    0), static_broadcasted_argnums=(0, 5, 6, 7, 8))
def _p_generate(pipe, prompt_ids, image, params, prng_seed, start_timestep,
    num_inference_steps, height, width, guidance_scale, noise, neg_prompt_ids):
    return pipe._generate(prompt_ids, image, params, prng_seed,
        start_timestep, num_inference_steps, height, width, guidance_scale,
        noise, neg_prompt_ids)
