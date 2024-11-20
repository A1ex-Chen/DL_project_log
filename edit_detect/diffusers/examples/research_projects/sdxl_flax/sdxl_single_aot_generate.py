def generate(prompt, negative_prompt, seed=default_seed, guidance_scale=
    default_guidance_scale):
    prompt_ids, neg_prompt_ids = tokenize_prompt(prompt, negative_prompt)
    prompt_ids, neg_prompt_ids, rng = replicate_all(prompt_ids,
        neg_prompt_ids, seed)
    g = jnp.array([guidance_scale] * prompt_ids.shape[0], dtype=jnp.float32)
    g = g[:, None]
    images = p_generate(prompt_ids, p_params, rng, g, None, neg_prompt_ids)
    images = images.reshape((images.shape[0] * images.shape[1],) + images.
        shape[-3:])
    return pipeline.numpy_to_pil(np.array(images))
