def test_stable_diffusion_k_euler(self):
    device = 'cpu'
    components = self.get_dummy_components()
    components['scheduler'] = EulerDiscreteScheduler.from_config(components
        ['scheduler'].config)
    sd_pipe = StableDiffusionPipeline(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = sd_pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 64, 64, 3)
    expected_slice = np.array([0.4865, 0.5439, 0.484, 0.4995, 0.5543, 
        0.4846, 0.5199, 0.4942, 0.5061])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
