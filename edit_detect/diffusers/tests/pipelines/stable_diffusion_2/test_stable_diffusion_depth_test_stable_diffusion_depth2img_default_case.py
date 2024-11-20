def test_stable_diffusion_depth2img_default_case(self):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = StableDiffusionDepth2ImgPipeline(**components)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 32, 32, 3)
    if torch_device == 'mps':
        expected_slice = np.array([0.6071, 0.5035, 0.4378, 0.5776, 0.5753, 
            0.4316, 0.4513, 0.5263, 0.4546])
    else:
        expected_slice = np.array([0.5435, 0.4992, 0.3783, 0.4411, 0.5842, 
            0.4654, 0.3786, 0.5077, 0.4655])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
