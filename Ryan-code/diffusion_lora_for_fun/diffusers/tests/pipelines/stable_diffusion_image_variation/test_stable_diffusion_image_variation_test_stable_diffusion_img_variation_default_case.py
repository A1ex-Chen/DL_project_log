def test_stable_diffusion_img_variation_default_case(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionImageVariationPipeline(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = sd_pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 64, 64, 3)
    expected_slice = np.array([0.5239, 0.5723, 0.4796, 0.5049, 0.555, 
        0.4685, 0.5329, 0.4891, 0.4921])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
