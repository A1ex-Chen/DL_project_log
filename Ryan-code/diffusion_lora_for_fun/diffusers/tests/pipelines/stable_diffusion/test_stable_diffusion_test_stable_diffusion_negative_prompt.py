def test_stable_diffusion_negative_prompt(self):
    device = 'cpu'
    components = self.get_dummy_components()
    components['scheduler'] = PNDMScheduler(skip_prk_steps=True)
    sd_pipe = StableDiffusionPipeline(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    negative_prompt = 'french fries'
    output = sd_pipe(**inputs, negative_prompt=negative_prompt)
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 64, 64, 3)
    expected_slice = np.array([0.1907, 0.4709, 0.4858, 0.2224, 0.4223, 
        0.4539, 0.5606, 0.3489, 0.39])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
