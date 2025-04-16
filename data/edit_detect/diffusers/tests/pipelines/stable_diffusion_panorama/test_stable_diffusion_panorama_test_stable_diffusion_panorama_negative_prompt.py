def test_stable_diffusion_panorama_negative_prompt(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionPanoramaPipeline(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    negative_prompt = 'french fries'
    output = sd_pipe(**inputs, negative_prompt=negative_prompt)
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 64, 64, 3)
    expected_slice = np.array([0.6187, 0.5375, 0.4915, 0.4136, 0.4114, 
        0.4563, 0.5128, 0.4976, 0.4757])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
