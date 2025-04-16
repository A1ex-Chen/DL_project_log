def test_stable_diffusion_img2img_tiny_autoencoder(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionImg2ImgPipeline(**components)
    sd_pipe.vae = self.get_dummy_tiny_autoencoder()
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = sd_pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 32, 32, 3)
    expected_slice = np.array([0.00669, 0.00669, 0.0, 0.00693, 0.00858, 0.0,
        0.00567, 0.00515, 0.00125])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
