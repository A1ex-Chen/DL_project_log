def test_inference_default_ddpm(self):
    init_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/img2img/sketch-mountains-input.jpg'
        )
    init_image = init_image.resize((128, 128))
    pipe = OnnxStableDiffusionUpscalePipeline.from_pretrained(
        'ssube/stable-diffusion-x4-upscaler-onnx', provider=self.
        gpu_provider, sess_options=self.gpu_options)
    pipe.set_progress_bar_config(disable=None)
    prompt = 'A fantasy landscape, trending on artstation'
    generator = np.random.RandomState(0)
    output = pipe(prompt=prompt, image=init_image, guidance_scale=7.5,
        num_inference_steps=10, generator=generator, output_type='np')
    images = output.images
    image_slice = images[0, 255:258, 383:386, -1]
    assert images.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.4883, 0.4947, 0.498, 0.4975, 0.4982, 0.498,
        0.5, 0.5006, 0.4972])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.02
