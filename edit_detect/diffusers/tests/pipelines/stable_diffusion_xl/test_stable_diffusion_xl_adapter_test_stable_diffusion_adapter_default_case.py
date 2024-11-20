def test_stable_diffusion_adapter_default_case(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionXLAdapterPipeline(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = sd_pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 64, 64, 3)
    expected_slice = np.array([0.5813032, 0.60995954, 0.47563356, 0.5056669,
        0.57199144, 0.4631841, 0.5176794, 0.51252556, 0.47183886])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.005
