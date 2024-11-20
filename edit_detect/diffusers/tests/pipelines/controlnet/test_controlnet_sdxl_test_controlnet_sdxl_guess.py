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
    expected_slice = np.array([0.6831671, 0.5702532, 0.5459845, 0.6299793, 
        0.58563006, 0.6033695, 0.4493941, 0.46132287, 0.5035841])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.0001
