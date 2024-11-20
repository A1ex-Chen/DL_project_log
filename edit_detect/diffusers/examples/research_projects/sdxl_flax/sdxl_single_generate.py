def generate(prompt, negative_prompt, seed=default_seed, guidance_scale=
    default_guidance_scale, num_inference_steps=default_num_steps):
    prompt_ids, neg_prompt_ids = tokenize_prompt(prompt, negative_prompt)
    prompt_ids, neg_prompt_ids, rng = replicate_all(prompt_ids,
        neg_prompt_ids, seed)
    images = pipeline(prompt_ids, p_params, rng, num_inference_steps=
        num_inference_steps, neg_prompt_ids=neg_prompt_ids, guidance_scale=
        guidance_scale, jit=True).images
    images = images.reshape((images.shape[0] * images.shape[1],) + images.
        shape[-3:])
    return pipeline.numpy_to_pil(np.array(images))
