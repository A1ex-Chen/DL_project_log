def test_inference(self):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    self.assertEqual(image.shape, (1, 256, 256, 3))
    expected_slice = np.array([0.47222412, 0.41921633, 0.44717434, 
        0.46874192, 0.42588258, 0.46150726, 0.4677534, 0.45583832, 0.48579055])
    max_diff = np.abs(image_slice.flatten() - expected_slice).max()
    self.assertLessEqual(max_diff, 0.001)
