def test_dummy_all_tpus(self):
    pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
        'hf-internal-testing/tiny-stable-diffusion-pipe', safety_checker=None)
    prompt = (
        'A cinematic film still of Morgan Freeman starring as Jimi Hendrix, portrait, 40mm lens, shallow depth of field, close up, split lighting, cinematic'
        )
    prng_seed = jax.random.PRNGKey(0)
    num_inference_steps = 4
    num_samples = jax.device_count()
    prompt = num_samples * [prompt]
    prompt_ids = pipeline.prepare_inputs(prompt)
    params = replicate(params)
    prng_seed = jax.random.split(prng_seed, num_samples)
    prompt_ids = shard(prompt_ids)
    images = pipeline(prompt_ids, params, prng_seed, num_inference_steps,
        jit=True).images
    assert images.shape == (num_samples, 1, 64, 64, 3)
    if jax.device_count() == 8:
        assert np.abs(np.abs(images[0, 0, :2, :2, -2:], dtype=np.float32).
            sum() - 4.1514745) < 0.001
        assert np.abs(np.abs(images, dtype=np.float32).sum() - 49947.875) < 0.5
    images_pil = pipeline.numpy_to_pil(np.asarray(images.reshape((
        num_samples,) + images.shape[-3:])))
    assert len(images_pil) == num_samples
