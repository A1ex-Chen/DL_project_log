def test_inference_k_lms(self):
    init_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/img2img/sketch-mountains-input.jpg'
        )
    init_image = init_image.resize((768, 512))
    lms_scheduler = LMSDiscreteScheduler.from_pretrained(
        'runwayml/stable-diffusion-v1-5', subfolder='scheduler', revision=
        'onnx')
    pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', revision='onnx', scheduler=
        lms_scheduler, safety_checker=None, feature_extractor=None,
        provider=self.gpu_provider, sess_options=self.gpu_options)
    pipe.set_progress_bar_config(disable=None)
    prompt = 'A fantasy landscape, trending on artstation'
    generator = np.random.RandomState(0)
    output = pipe(prompt=prompt, image=init_image, strength=0.75,
        guidance_scale=7.5, num_inference_steps=20, generator=generator,
        output_type='np')
    images = output.images
    image_slice = images[0, 255:258, 383:386, -1]
    assert images.shape == (1, 512, 768, 3)
    expected_slice = np.array([0.8043, 0.926, 0.9581, 0.8119, 0.8954, 0.913,
        0.7209, 0.7463, 0.7431])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.02
