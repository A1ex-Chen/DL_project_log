def test_jax_memory_efficient_attention(self):
    prompt = (
        'A cinematic film still of Morgan Freeman starring as Jimi Hendrix, portrait, 40mm lens, shallow depth of field, close up, split lighting, cinematic'
        )
    num_samples = jax.device_count()
    prompt = num_samples * [prompt]
    prng_seed = jax.random.split(jax.random.PRNGKey(0), num_samples)
    pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', revision='bf16', dtype=jnp.
        bfloat16, safety_checker=None)
    params = replicate(params)
    prompt_ids = pipeline.prepare_inputs(prompt)
    prompt_ids = shard(prompt_ids)
    images = pipeline(prompt_ids, params, prng_seed, jit=True).images
    assert images.shape == (num_samples, 1, 512, 512, 3)
    slice = images[2, 0, 256, 10:17, 1]
    pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', revision='bf16', dtype=jnp.
        bfloat16, safety_checker=None, use_memory_efficient_attention=True)
    params = replicate(params)
    prompt_ids = pipeline.prepare_inputs(prompt)
    prompt_ids = shard(prompt_ids)
    images_eff = pipeline(prompt_ids, params, prng_seed, jit=True).images
    assert images_eff.shape == (num_samples, 1, 512, 512, 3)
    slice_eff = images[2, 0, 256, 10:17, 1]
    assert abs(slice_eff - slice).max() < 0.01
