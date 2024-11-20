def test_kandinsky_prior_emb2emb(self):
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
    expected_slice = np.array([0.1071284, 1.3330271, 0.61260223, -0.6691065,
        -0.3846852, -1.0303661, 0.22716111, 0.03348901, 0.30040675, -
        0.24805029])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
    assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max(
        ) < 0.01
