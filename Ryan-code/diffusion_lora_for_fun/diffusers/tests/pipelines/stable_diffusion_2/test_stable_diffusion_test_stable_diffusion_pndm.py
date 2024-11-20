def test_stable_diffusion_pndm(self):
    pipe = StableDiffusionPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-base')
    pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs(torch_device)
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1].flatten()
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.49493, 0.47896, 0.40798, 0.54214, 0.53212,
        0.48202, 0.47656, 0.46329, 0.48506])
    assert np.abs(image_slice - expected_slice).max() < 0.007
