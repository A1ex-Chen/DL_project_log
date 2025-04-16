def test_stable_diffusion_k_euler_ancestral(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionPipeline(**components)
    sd_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(sd_pipe
        .scheduler.config)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    output = sd_pipe(**inputs)
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 64, 64, 3)
    expected_slice = np.array([0.2682, 0.4782, 0.4855, 0.2424, 0.4472, 
        0.4479, 0.5612, 0.3676, 0.3854])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
