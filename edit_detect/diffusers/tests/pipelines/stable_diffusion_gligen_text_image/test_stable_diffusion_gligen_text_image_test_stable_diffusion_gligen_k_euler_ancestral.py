def test_stable_diffusion_gligen_k_euler_ancestral(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionGLIGENTextImagePipeline(**components)
    sd_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(sd_pipe
        .scheduler.config)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = sd_pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 64, 64, 3)
    expected_slice = np.array([0.425, 0.494, 0.429, 0.469, 0.525, 0.417, 
        0.533, 0.5, 0.47])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
