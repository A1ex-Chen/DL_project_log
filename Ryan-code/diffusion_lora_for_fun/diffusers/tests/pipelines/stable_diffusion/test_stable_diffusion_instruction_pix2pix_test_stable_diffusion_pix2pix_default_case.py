def test_stable_diffusion_pix2pix_default_case(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionInstructPix2PixPipeline(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = sd_pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 32, 32, 3)
    expected_slice = np.array([0.7526, 0.375, 0.4547, 0.6117, 0.5866, 
        0.5016, 0.4327, 0.5642, 0.4815])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
