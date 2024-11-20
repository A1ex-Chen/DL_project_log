def test_stable_diffusion_depth2img_multiple_init_images(self):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = StableDiffusionDepth2ImgPipeline(**components)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    inputs['prompt'] = [inputs['prompt']] * 2
    inputs['image'] = 2 * [inputs['image']]
    image = pipe(**inputs).images
    image_slice = image[-1, -3:, -3:, -1]
    assert image.shape == (2, 32, 32, 3)
    if torch_device == 'mps':
        expected_slice = np.array([0.6501, 0.515, 0.4939, 0.6688, 0.5437, 
            0.5758, 0.5115, 0.4406, 0.4551])
    else:
        expected_slice = np.array([0.6557, 0.6214, 0.6254, 0.5775, 0.4785, 
            0.5949, 0.5904, 0.4785, 0.473])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
