def test_inference_non_square_images(self):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = pipe(**inputs, height=32, width=48).images
    image_slice = image[0, -3:, -3:, -1]
    self.assertEqual(image.shape, (1, 32, 48, 3))
    expected_slice = np.array([0.6493, 0.537, 0.4081, 0.4762, 0.3695, 
        0.4711, 0.3026, 0.5218, 0.5263])
    max_diff = np.abs(image_slice.flatten() - expected_slice).max()
    self.assertLessEqual(max_diff, 0.001)
