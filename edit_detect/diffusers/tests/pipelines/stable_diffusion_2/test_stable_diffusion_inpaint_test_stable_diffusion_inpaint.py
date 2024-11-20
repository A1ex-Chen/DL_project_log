def test_stable_diffusion_inpaint(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionInpaintPipeline(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = sd_pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 64, 64, 3)
    expected_slice = np.array([0.4727, 0.5735, 0.3941, 0.5446, 0.5926, 
        0.4394, 0.5062, 0.4654, 0.4476])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
