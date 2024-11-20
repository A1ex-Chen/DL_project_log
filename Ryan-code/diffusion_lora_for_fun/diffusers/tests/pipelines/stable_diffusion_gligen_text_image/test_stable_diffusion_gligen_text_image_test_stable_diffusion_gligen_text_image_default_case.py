def test_stable_diffusion_gligen_text_image_default_case(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionGLIGENTextImagePipeline(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = sd_pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 64, 64, 3)
    expected_slice = np.array([0.5069, 0.5561, 0.4577, 0.4792, 0.5203, 
        0.4089, 0.5039, 0.4919, 0.4499])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
