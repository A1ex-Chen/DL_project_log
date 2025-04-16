def test_stable_diffusion_ddim_factor_8(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionPipeline(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    output = sd_pipe(**inputs, height=136, width=136)
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 136, 136, 3)
    expected_slice = np.array([0.472, 0.5426, 0.516, 0.3961, 0.4696, 0.4296,
        0.5738, 0.5888, 0.5481])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
