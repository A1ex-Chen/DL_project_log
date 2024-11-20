def test_stable_diffusion_depth2img_negative_prompt(self):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = StableDiffusionDepth2ImgPipeline(**components)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    negative_prompt = 'french fries'
    output = pipe(**inputs, negative_prompt=negative_prompt)
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 32, 32, 3)
    if torch_device == 'mps':
        expected_slice = np.array([0.6296, 0.5125, 0.389, 0.4456, 0.5955, 
            0.4621, 0.381, 0.531, 0.4626])
    else:
        expected_slice = np.array([0.6012, 0.4507, 0.3769, 0.4121, 0.5566, 
            0.4585, 0.3803, 0.5045, 0.4631])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
