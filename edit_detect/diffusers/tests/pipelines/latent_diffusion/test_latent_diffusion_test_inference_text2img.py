def test_inference_text2img(self):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = LDMTextToImagePipeline(**components)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 16, 16, 3)
    expected_slice = np.array([0.6101, 0.6156, 0.5622, 0.4895, 0.6661, 
        0.3804, 0.5748, 0.6136, 0.5014])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
