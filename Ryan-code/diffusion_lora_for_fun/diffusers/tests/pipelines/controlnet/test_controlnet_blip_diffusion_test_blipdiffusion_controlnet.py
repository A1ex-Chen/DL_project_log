def test_blipdiffusion_controlnet(self):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    image = pipe(**self.get_dummy_inputs(device))[0]
    image_slice = image[0, -3:, -3:, 0]
    assert image.shape == (1, 16, 16, 4)
    expected_slice = np.array([0.7953, 0.7136, 0.6597, 0.4779, 0.7389, 
        0.4111, 0.5826, 0.415, 0.8422])
    assert np.abs(image_slice.flatten() - expected_slice).max(
        ) < 0.01, f' expected_slice {expected_slice}, but got {image_slice.flatten()}'
