def test_inference_k_lms(self):
    init_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/img2img/sketch-mountains-input.jpg'
        )
    init_image = init_image.resize((128, 128))
    lms_scheduler = LMSDiscreteScheduler.from_pretrained(
        'ssube/stable-diffusion-x4-upscaler-onnx', subfolder='scheduler')
    pipe = OnnxStableDiffusionUpscalePipeline.from_pretrained(
        'ssube/stable-diffusion-x4-upscaler-onnx', scheduler=lms_scheduler,
        provider=self.gpu_provider, sess_options=self.gpu_options)
    pipe.set_progress_bar_config(disable=None)
    prompt = 'A fantasy landscape, trending on artstation'
    generator = np.random.RandomState(0)
    output = pipe(prompt=prompt, image=init_image, guidance_scale=7.5,
        num_inference_steps=20, generator=generator, output_type='np')
    images = output.images
    image_slice = images[0, 255:258, 383:386, -1]
    assert images.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.50173753, 0.50223356, 0.502039, 0.50233036,
        0.5023725, 0.5022601, 0.5018758, 0.50234085, 0.50241566])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.02
