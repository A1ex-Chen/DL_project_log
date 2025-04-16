def test_kandinsky(self):
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
    expected_slice = np.array([0.342, 0.9505, 0.3919, 1.0, 0.5188, 0.3109, 
        0.6139, 0.5624, 0.6811])
    assert np.abs(image_slice.flatten() - expected_slice).max(
        ) < 0.01, f' expected_slice {expected_slice}, but got {image_slice.flatten()}'
    assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max(
        ) < 0.01, f' expected_slice {expected_slice}, but got {image_from_tuple_slice.flatten()}'
