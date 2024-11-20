def test_kandinsky3(self):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    output = pipe(**self.get_dummy_inputs(device))
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 16, 16, 3)
    expected_slice = np.array([0.3768, 0.4373, 0.4865, 0.489, 0.4299, 
        0.5122, 0.4921, 0.4924, 0.5599])
    assert np.abs(image_slice.flatten() - expected_slice).max(
        ) < 0.01, f' expected_slice {expected_slice}, but got {image_slice.flatten()}'
