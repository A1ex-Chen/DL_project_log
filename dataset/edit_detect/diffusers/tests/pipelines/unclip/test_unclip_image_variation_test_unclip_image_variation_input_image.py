def test_unclip_image_variation_input_image(self):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    pipeline_inputs = self.get_dummy_inputs(device, pil_image=True)
    output = pipe(**pipeline_inputs)
    image = output.images
    tuple_pipeline_inputs = self.get_dummy_inputs(device, pil_image=True)
    image_from_tuple = pipe(**tuple_pipeline_inputs, return_dict=False)[0]
    image_slice = image[0, -3:, -3:, -1]
    image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]
    assert image.shape == (1, 64, 64, 3)
    expected_slice = np.array([0.9997, 0.0003, 0.9997, 0.9997, 0.997, 
        0.0024, 0.9997, 0.9971, 0.9971])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
    assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max(
        ) < 0.01
