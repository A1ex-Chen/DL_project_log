def test_lcm_custom_timesteps(self):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = LatentConsistencyModelPipeline(**components)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    del inputs['num_inference_steps']
    inputs['timesteps'] = [999, 499]
    output = pipe(**inputs)
    image = output.images
    assert image.shape == (1, 64, 64, 3)
    image_slice = image[0, -3:, -3:, -1]
    expected_slice = np.array([0.1403, 0.5072, 0.5316, 0.1202, 0.3865, 
        0.4211, 0.5363, 0.3557, 0.3645])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
