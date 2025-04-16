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
    expected_slice = np.array([0.6319, 0.3526, 0.3806, 0.6327, 0.4639, 
        0.483, 0.2583, 0.5331, 0.4852])
    max_diff = np.abs(image_slice.flatten() - expected_slice).max()
    self.assertLessEqual(max_diff, 0.001)
