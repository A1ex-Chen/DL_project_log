def test_lcm_custom_timesteps(self):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    del inputs['num_inference_steps']
    inputs['timesteps'] = [999, 499]
    output = pipe(**inputs)
    image = output.images
    assert image.shape == (1, 32, 32, 3)
    image_slice = image[0, -3:, -3:, -1]
    expected_slice = np.array([0.3994, 0.3471, 0.254, 0.703, 0.6193, 0.3645,
        0.5777, 0.585, 0.4965])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
