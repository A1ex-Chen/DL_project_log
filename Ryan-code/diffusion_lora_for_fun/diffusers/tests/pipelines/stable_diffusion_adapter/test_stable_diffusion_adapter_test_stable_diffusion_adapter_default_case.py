def test_stable_diffusion_adapter_default_case(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionAdapterPipeline(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = sd_pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 64, 64, 3)
    expected_slice = np.array([0.4902, 0.5539, 0.4317, 0.4682, 0.619, 
        0.4351, 0.5018, 0.5046, 0.4772])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.005
