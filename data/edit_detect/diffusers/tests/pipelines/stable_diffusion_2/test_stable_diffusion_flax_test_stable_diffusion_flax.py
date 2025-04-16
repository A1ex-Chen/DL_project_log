def test_stable_diffusion_flax(self):
    sd_pipe, params = FlaxStableDiffusionPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2', revision='bf16', dtype=jnp.bfloat16)
    prompt = 'A painting of a squirrel eating a burger'
    num_samples = jax.device_count()
    prompt = num_samples * [prompt]
    prompt_ids = sd_pipe.prepare_inputs(prompt)
    params = replicate(params)
    prompt_ids = shard(prompt_ids)
    prng_seed = jax.random.PRNGKey(0)
    prng_seed = jax.random.split(prng_seed, jax.device_count())
    images = sd_pipe(prompt_ids, params, prng_seed, num_inference_steps=25,
        jit=True)[0]
    assert images.shape == (jax.device_count(), 1, 768, 768, 3)
    images = images.reshape((images.shape[0] * images.shape[1],) + images.
        shape[-3:])
    image_slice = images[0, 253:256, 253:256, -1]
    output_slice = jnp.asarray(jax.device_get(image_slice.flatten()))
    expected_slice = jnp.array([0.4238, 0.4414, 0.4395, 0.4453, 0.4629, 
        0.459, 0.4531, 0.45508, 0.4512])
    print(f'output_slice: {output_slice}')
    assert jnp.abs(output_slice - expected_slice).max() < 0.01
