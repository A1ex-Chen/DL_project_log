def test_stable_diffusion_xl_img2img_tiny_autoencoder(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionXLImg2ImgPipeline(**components)
    sd_pipe.vae = self.get_dummy_tiny_autoencoder()
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = sd_pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1].flatten()
    assert image.shape == (1, 32, 32, 3)
    expected_slice = np.array([0.0, 0.0, 0.0106, 0.0, 0.0, 0.0087, 0.0052, 
        0.0062, 0.0177])
    assert np.allclose(image_slice, expected_slice, atol=0.0001, rtol=0.0001)
