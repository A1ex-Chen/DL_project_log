def test_stable_diffusion_img2img_default_case(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionImg2ImgPipeline(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = sd_pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 32, 32, 3)
    expected_slice = np.array([0.4555, 0.3216, 0.4049, 0.462, 0.4618, 
        0.4126, 0.4122, 0.4629, 0.4579])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
