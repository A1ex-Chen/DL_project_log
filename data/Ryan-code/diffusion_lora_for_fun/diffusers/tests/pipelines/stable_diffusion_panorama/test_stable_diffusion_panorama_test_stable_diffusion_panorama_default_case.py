def test_stable_diffusion_panorama_default_case(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionPanoramaPipeline(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = sd_pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 64, 64, 3)
    expected_slice = np.array([0.6186, 0.5374, 0.4915, 0.4135, 0.4114, 
        0.4563, 0.5128, 0.4977, 0.4757])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
