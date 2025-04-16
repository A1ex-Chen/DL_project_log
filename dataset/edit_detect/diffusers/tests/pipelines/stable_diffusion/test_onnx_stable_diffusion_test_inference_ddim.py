def test_inference_ddim(self):
    ddim_scheduler = DDIMScheduler.from_pretrained(
        'runwayml/stable-diffusion-v1-5', subfolder='scheduler', revision=
        'onnx')
    sd_pipe = OnnxStableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', revision='onnx', scheduler=
        ddim_scheduler, safety_checker=None, feature_extractor=None,
        provider=self.gpu_provider, sess_options=self.gpu_options)
    sd_pipe.set_progress_bar_config(disable=None)
    prompt = 'open neural network exchange'
    generator = np.random.RandomState(0)
    output = sd_pipe([prompt], guidance_scale=7.5, num_inference_steps=10,
        generator=generator, output_type='np')
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.2867, 0.1974, 0.1481, 0.7294, 0.7251, 
        0.6667, 0.4194, 0.5642, 0.6486])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
