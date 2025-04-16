def test_canny(self):
    controlnet, controlnet_params = FlaxControlNetModel.from_pretrained(
        'lllyasviel/sd-controlnet-canny', from_pt=True, dtype=jnp.bfloat16)
    pipe, params = FlaxStableDiffusionControlNetPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', controlnet=controlnet, from_pt=
        True, dtype=jnp.bfloat16)
    params['controlnet'] = controlnet_params
    prompts = 'bird'
    num_samples = jax.device_count()
    prompt_ids = pipe.prepare_text_inputs([prompts] * num_samples)
    canny_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png'
        )
    processed_image = pipe.prepare_image_inputs([canny_image] * num_samples)
    rng = jax.random.PRNGKey(0)
    rng = jax.random.split(rng, jax.device_count())
    p_params = replicate(params)
    prompt_ids = shard(prompt_ids)
    processed_image = shard(processed_image)
    images = pipe(prompt_ids=prompt_ids, image=processed_image, params=
        p_params, prng_seed=rng, num_inference_steps=50, jit=True).images
    assert images.shape == (jax.device_count(), 1, 768, 512, 3)
    images = images.reshape((images.shape[0] * images.shape[1],) + images.
        shape[-3:])
    image_slice = images[0, 253:256, 253:256, -1]
    output_slice = jnp.asarray(jax.device_get(image_slice.flatten()))
    expected_slice = jnp.array([0.167969, 0.116699, 0.081543, 0.154297, 
        0.132812, 0.108887, 0.169922, 0.169922, 0.205078])
    print(f'output_slice: {output_slice}')
    assert jnp.abs(output_slice - expected_slice).max() < 0.01
