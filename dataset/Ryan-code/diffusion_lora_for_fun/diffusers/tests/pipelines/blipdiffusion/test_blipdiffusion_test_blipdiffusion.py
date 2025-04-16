def test_blipdiffusion(self):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    image = pipe(**self.get_dummy_inputs(device))[0]
    image_slice = image[0, -3:, -3:, 0]
    assert image.shape == (1, 16, 16, 4)
    expected_slice = np.array([0.5329548, 0.8372512, 0.33269387, 0.82096875,
        0.43657133, 0.3783, 0.5953028, 0.51934963, 0.42142007])
    assert np.abs(image_slice.flatten() - expected_slice).max(
        ) < 0.01, f' expected_slice {image_slice.flatten()}, but got {image_slice.flatten()}'
