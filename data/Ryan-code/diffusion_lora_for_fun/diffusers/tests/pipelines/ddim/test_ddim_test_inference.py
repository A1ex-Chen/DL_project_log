def test_inference(self):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    self.assertEqual(image.shape, (1, 8, 8, 3))
    expected_slice = np.array([0.0, 0.9979, 0.0, 0.9999, 0.9986, 0.9991, 
        0.0007106, 0.0, 0.0])
    max_diff = np.abs(image_slice.flatten() - expected_slice).max()
    self.assertLessEqual(max_diff, 0.001)
