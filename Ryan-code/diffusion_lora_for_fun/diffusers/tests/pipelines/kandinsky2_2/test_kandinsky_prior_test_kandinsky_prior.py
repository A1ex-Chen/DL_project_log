def test_kandinsky_prior(self):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    output = pipe(**self.get_dummy_inputs(device))
    image = output.image_embeds
    image_from_tuple = pipe(**self.get_dummy_inputs(device), return_dict=False
        )[0]
    image_slice = image[0, -10:]
    image_from_tuple_slice = image_from_tuple[0, -10:]
    assert image.shape == (1, 32)
    expected_slice = np.array([-0.0532, 1.712, 0.3656, -1.0852, -0.8946, -
        1.1756, 0.4348, 0.2482, 0.5146, -0.1156])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
    assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max(
        ) < 0.01
