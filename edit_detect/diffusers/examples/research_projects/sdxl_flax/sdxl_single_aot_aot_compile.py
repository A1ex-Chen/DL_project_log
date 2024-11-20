def aot_compile(prompt=default_prompt, negative_prompt=default_neg_prompt,
    seed=default_seed, guidance_scale=default_guidance_scale,
    num_inference_steps=default_num_steps):
    prompt_ids, neg_prompt_ids = tokenize_prompt(prompt, negative_prompt)
    prompt_ids, neg_prompt_ids, rng = replicate_all(prompt_ids,
        neg_prompt_ids, seed)
    g = jnp.array([guidance_scale] * prompt_ids.shape[0], dtype=jnp.float32)
    g = g[:, None]
    return pmap(pipeline._generate, static_broadcasted_argnums=[3, 4, 5, 9]
        ).lower(prompt_ids, p_params, rng, num_inference_steps, height,
        width, g, None, neg_prompt_ids, False).compile()
