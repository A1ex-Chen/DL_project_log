def test_inference(self):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    self.assertEqual(image.shape, (1, 16, 16, 3))
    expected_slice = np.array([0.2946, 0.6601, 0.4329, 0.3296, 0.4144, 
        0.5319, 0.7273, 0.5013, 0.4457])
    max_diff = np.abs(image_slice.flatten() - expected_slice).max()
    self.assertLessEqual(max_diff, 0.001)
