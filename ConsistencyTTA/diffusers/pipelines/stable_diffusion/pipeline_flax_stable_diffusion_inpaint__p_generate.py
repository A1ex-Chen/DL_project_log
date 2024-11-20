@partial(jax.pmap, in_axes=(None, 0, 0, 0, 0, 0, None, None, None, 0, 0, 0),
    static_broadcasted_argnums=(0, 6, 7, 8))
def _p_generate(pipe, prompt_ids, mask, masked_image, params, prng_seed,
    num_inference_steps, height, width, guidance_scale, latents, neg_prompt_ids
    ):
    return pipe._generate(prompt_ids, mask, masked_image, params, prng_seed,
        num_inference_steps, height, width, guidance_scale, latents,
        neg_prompt_ids)
