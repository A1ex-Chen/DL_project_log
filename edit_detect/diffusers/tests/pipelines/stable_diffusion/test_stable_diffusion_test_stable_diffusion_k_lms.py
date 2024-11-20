def test_stable_diffusion_k_lms(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionPipeline(**components)
    sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.
        config)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    output = sd_pipe(**inputs)
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 64, 64, 3)
    expected_slice = np.array([0.2681, 0.4785, 0.4857, 0.2426, 0.4473, 
        0.4481, 0.561, 0.3676, 0.3855])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
