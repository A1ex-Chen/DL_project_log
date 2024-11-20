def test_stable_diffusion_pndm(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionPipeline(**components)
    sd_pipe.scheduler = PNDMScheduler(skip_prk_steps=True)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    output = sd_pipe(**inputs)
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 64, 64, 3)
    expected_slice = np.array([0.1941, 0.4748, 0.488, 0.2222, 0.4221, 
        0.4545, 0.5604, 0.3488, 0.3902])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
