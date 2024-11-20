def test_controlnet_sdxl_guess(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = self.pipeline_class(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    inputs['guess_mode'] = True
    output = sd_pipe(**inputs)
    image_slice = output.images[0, -3:, -3:, -1]
    expected_slice = np.array([0.549, 0.5053, 0.4676, 0.5816, 0.5364, 0.483,
        0.5937, 0.5719, 0.4318])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.0001
