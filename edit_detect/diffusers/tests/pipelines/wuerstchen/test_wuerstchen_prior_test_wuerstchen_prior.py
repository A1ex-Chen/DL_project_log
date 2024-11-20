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
    assert image.shape == (1, 2, 24, 24)
    expected_slice = np.array([-7172.837, -3438.855, -1093.312, 388.8835, -
        7471.467, -7998.1206, -5328.259, 218.00089, -2731.5745, -8056.734])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.05
    assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max(
        ) < 0.05
