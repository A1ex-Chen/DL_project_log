def test_inference_default_pndm(self):
    init_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/in_paint/overture-creations-5sI6fQgYIuo.png'
        )
    mask_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/in_paint/overture-creations-5sI6fQgYIuo_mask.png'
        )
    pipe = OnnxStableDiffusionInpaintPipeline.from_pretrained(
        'runwayml/stable-diffusion-inpainting', revision='onnx',
        safety_checker=None, feature_extractor=None, provider=self.
        gpu_provider, sess_options=self.gpu_options)
    pipe.set_progress_bar_config(disable=None)
    prompt = 'A red cat sitting on a park bench'
    generator = np.random.RandomState(0)
    output = pipe(prompt=prompt, image=init_image, mask_image=mask_image,
        guidance_scale=7.5, num_inference_steps=10, generator=generator,
        output_type='np')
    images = output.images
    image_slice = images[0, 255:258, 255:258, -1]
    assert images.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.2514, 0.3007, 0.3517, 0.179, 0.2382, 
        0.3167, 0.1944, 0.2273, 0.2464])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
