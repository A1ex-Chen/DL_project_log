def test_stable_diffusion_dpm_flax(self):
    model_id = 'stabilityai/stable-diffusion-2'
    scheduler, scheduler_params = (FlaxDPMSolverMultistepScheduler.
        from_pretrained(model_id, subfolder='scheduler'))
    sd_pipe, params = FlaxStableDiffusionPipeline.from_pretrained(model_id,
        scheduler=scheduler, revision='bf16', dtype=jnp.bfloat16)
    params['scheduler'] = scheduler_params
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
    expected_slice = jnp.array([0.4336, 0.42969, 0.4453, 0.4199, 0.4297, 
        0.4531, 0.4434, 0.4434, 0.4297])
    print(f'output_slice: {output_slice}')
    assert jnp.abs(output_slice - expected_slice).max() < 0.01
