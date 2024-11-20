def test_unclip(self):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    output = pipe(**self.get_dummy_inputs(device))
    image = output.images
    image_from_tuple = pipe(**self.get_dummy_inputs(device), return_dict=False
        )[0]
    image_slice = image[0, -3:, -3:, -1]
    image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]
    assert image.shape == (1, 64, 64, 3)
    expected_slice = np.array([0.9997, 0.9988, 0.0028, 0.9997, 0.9984, 
        0.9965, 0.0029, 0.9986, 0.0025])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
    assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max(
        ) < 0.01
