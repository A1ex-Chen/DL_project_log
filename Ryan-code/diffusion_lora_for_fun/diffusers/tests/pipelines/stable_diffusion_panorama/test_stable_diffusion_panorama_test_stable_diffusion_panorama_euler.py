def test_stable_diffusion_panorama_euler(self):
    device = 'cpu'
    components = self.get_dummy_components()
    components['scheduler'] = EulerAncestralDiscreteScheduler(beta_start=
        0.00085, beta_end=0.012, beta_schedule='scaled_linear')
    sd_pipe = StableDiffusionPanoramaPipeline(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = sd_pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 64, 64, 3)
    expected_slice = np.array([0.4024, 0.651, 0.4901, 0.5378, 0.5813, 
        0.5622, 0.4795, 0.4467, 0.4952])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
