def test_stable_diffusion_pix2pix_negative_prompt(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionInstructPix2PixPipeline(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    negative_prompt = 'french fries'
    output = sd_pipe(**inputs, negative_prompt=negative_prompt)
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 32, 32, 3)
    expected_slice = np.array([0.7511, 0.3642, 0.4553, 0.6236, 0.5797, 
        0.5013, 0.4343, 0.5611, 0.4831])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
