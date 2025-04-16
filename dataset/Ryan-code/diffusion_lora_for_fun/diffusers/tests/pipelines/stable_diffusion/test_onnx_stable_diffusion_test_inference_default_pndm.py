def test_inference_default_pndm(self):
    sd_pipe = OnnxStableDiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', revision='onnx', safety_checker=
        None, feature_extractor=None, provider=self.gpu_provider,
        sess_options=self.gpu_options)
    sd_pipe.set_progress_bar_config(disable=None)
    prompt = 'A painting of a squirrel eating a burger'
    np.random.seed(0)
    output = sd_pipe([prompt], guidance_scale=6.0, num_inference_steps=10,
        output_type='np')
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.0452, 0.039, 0.0087, 0.035, 0.0617, 0.0364,
        0.0544, 0.0523, 0.072])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
