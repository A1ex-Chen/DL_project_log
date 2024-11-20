def test_mask(self):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_mask_inputs(device)
    mask = pipe.generate_mask(**inputs)
    mask_slice = mask[0, -3:, -3:]
    self.assertEqual(mask.shape, (1, 16, 16))
    expected_slice = np.array([0] * 9)
    max_diff = np.abs(mask_slice.flatten() - expected_slice).max()
    self.assertLessEqual(max_diff, 0.001)
    self.assertEqual(mask[0, -3, -4], 0)
