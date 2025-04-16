def test_inference_k_lms(self):
    init_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/in_paint/overture-creations-5sI6fQgYIuo.png'
        )
    mask_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/in_paint/overture-creations-5sI6fQgYIuo_mask.png'
        )
    lms_scheduler = LMSDiscreteScheduler.from_pretrained(
        'runwayml/stable-diffusion-inpainting', subfolder='scheduler',
        revision='onnx')
    pipe = OnnxStableDiffusionInpaintPipeline.from_pretrained(
        'runwayml/stable-diffusion-inpainting', revision='onnx', scheduler=
        lms_scheduler, safety_checker=None, feature_extractor=None,
        provider=self.gpu_provider, sess_options=self.gpu_options)
    pipe.set_progress_bar_config(disable=None)
    prompt = 'A red cat sitting on a park bench'
    generator = np.random.RandomState(0)
    output = pipe(prompt=prompt, image=init_image, mask_image=mask_image,
        guidance_scale=7.5, num_inference_steps=20, generator=generator,
        output_type='np')
    images = output.images
    image_slice = images[0, 255:258, 255:258, -1]
    assert images.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.0086, 0.0077, 0.0083, 0.0093, 0.0107, 
        0.0139, 0.0094, 0.0097, 0.0125])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
