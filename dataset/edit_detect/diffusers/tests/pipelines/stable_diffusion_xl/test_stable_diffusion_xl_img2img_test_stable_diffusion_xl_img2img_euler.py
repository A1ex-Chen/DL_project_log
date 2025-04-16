def test_stable_diffusion_xl_img2img_euler(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionXLImg2ImgPipeline(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = sd_pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 32, 32, 3)
    expected_slice = np.array([0.4745, 0.4924, 0.4338, 0.6468, 0.5547, 
        0.4419, 0.5646, 0.5897, 0.5146])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
