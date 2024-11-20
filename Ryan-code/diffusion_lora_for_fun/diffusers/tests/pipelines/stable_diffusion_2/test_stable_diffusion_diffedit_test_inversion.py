def test_inversion(self):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inversion_inputs(device)
    image = pipe.invert(**inputs).images
    image_slice = image[0, -1, -3:, -3:]
    self.assertEqual(image.shape, (2, 32, 32, 3))
    expected_slice = np.array([0.516, 0.5115, 0.506, 0.5456, 0.4704, 0.506,
        0.5019, 0.4405, 0.4726])
    max_diff = np.abs(image_slice.flatten() - expected_slice).max()
    self.assertLessEqual(max_diff, 0.001)
