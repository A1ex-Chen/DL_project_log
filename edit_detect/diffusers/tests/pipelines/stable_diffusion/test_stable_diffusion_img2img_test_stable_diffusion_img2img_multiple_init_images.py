def test_stable_diffusion_img2img_multiple_init_images(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionImg2ImgPipeline(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    inputs['prompt'] = [inputs['prompt']] * 2
    inputs['image'] = inputs['image'].repeat(2, 1, 1, 1)
    image = sd_pipe(**inputs).images
    image_slice = image[-1, -3:, -3:, -1]
    assert image.shape == (2, 32, 32, 3)
    expected_slice = np.array([0.4241, 0.5576, 0.5711, 0.4792, 0.4311, 
        0.5952, 0.5827, 0.5138, 0.5109])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
