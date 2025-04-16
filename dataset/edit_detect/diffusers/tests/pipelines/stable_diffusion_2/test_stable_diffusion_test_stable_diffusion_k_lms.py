def test_stable_diffusion_k_lms(self):
    pipe = StableDiffusionPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-base')
    pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs(torch_device)
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1].flatten()
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.1044, 0.13115, 0.111, 0.10141, 0.1144, 
        0.07215, 0.11332, 0.09693, 0.10006])
    assert np.abs(image_slice - expected_slice).max() < 0.003
