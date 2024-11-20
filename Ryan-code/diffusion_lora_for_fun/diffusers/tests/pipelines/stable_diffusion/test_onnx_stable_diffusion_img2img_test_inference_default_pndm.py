def test_inference_default_pndm(self):
    init_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/img2img/sketch-mountains-input.jpg'
        )
    init_image = init_image.resize((768, 512))
    pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', revision='onnx', safety_checker=
        None, feature_extractor=None, provider=self.gpu_provider,
        sess_options=self.gpu_options)
    pipe.set_progress_bar_config(disable=None)
    prompt = 'A fantasy landscape, trending on artstation'
    generator = np.random.RandomState(0)
    output = pipe(prompt=prompt, image=init_image, strength=0.75,
        guidance_scale=7.5, num_inference_steps=10, generator=generator,
        output_type='np')
    images = output.images
    image_slice = images[0, 255:258, 383:386, -1]
    assert images.shape == (1, 512, 768, 3)
    expected_slice = np.array([0.4909, 0.5059, 0.5372, 0.4623, 0.4876, 
        0.5049, 0.482, 0.4956, 0.5019])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.02
