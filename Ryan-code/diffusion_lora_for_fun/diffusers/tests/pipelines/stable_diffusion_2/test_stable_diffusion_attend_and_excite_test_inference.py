def test_inference(self):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    self.assertEqual(image.shape, (1, 64, 64, 3))
    expected_slice = np.array([0.63905364, 0.62897307, 0.48599017, 
        0.5133624, 0.5550048, 0.45769516, 0.50326973, 0.5023139, 0.45384496])
    max_diff = np.abs(image_slice.flatten() - expected_slice).max()
    self.assertLessEqual(max_diff, 0.001)
