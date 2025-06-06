def test_stable_diffusion_v1_4_bfloat_16_with_safety(self):
    pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', revision='bf16', dtype=jnp.bfloat16)
    prompt = (
        'A cinematic film still of Morgan Freeman starring as Jimi Hendrix, portrait, 40mm lens, shallow depth of field, close up, split lighting, cinematic'
        )
    prng_seed = jax.random.PRNGKey(0)
    num_inference_steps = 50
    num_samples = jax.device_count()
    prompt = num_samples * [prompt]
    prompt_ids = pipeline.prepare_inputs(prompt)
    params = replicate(params)
    prng_seed = jax.random.split(prng_seed, num_samples)
    prompt_ids = shard(prompt_ids)
    images = pipeline(prompt_ids, params, prng_seed, num_inference_steps,
        jit=True).images
    assert images.shape == (num_samples, 1, 512, 512, 3)
    if jax.device_count() == 8:
        assert np.abs(np.abs(images[0, 0, :2, :2, -2:], dtype=np.float32).
            sum() - 0.04003906) < 0.05
        assert np.abs(np.abs(images, dtype=np.float32).sum() - 2373516.75
            ) < 0.5
