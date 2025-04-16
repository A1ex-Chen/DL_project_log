def test_wuerstchen_prior(self):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    output = pipe(**self.get_dummy_inputs(device))
    image = output.image_embeddings
    image_from_tuple = pipe(**self.get_dummy_inputs(device), return_dict=False
        )[0]
    image_slice = image[0, 0, 0, -10:]
    image_from_tuple_slice = image_from_tuple[0, 0, 0, -10:]
    assert image.shape == (1, 16, 24, 24)
    expected_slice = np.array([96.139565, -20.213179, -116.40341, -
        191.57129, 39.350136, 74.80767, 39.782352, -184.67352, -46.426907, 
        168.41783])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.05
    assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max(
        ) < 0.05
