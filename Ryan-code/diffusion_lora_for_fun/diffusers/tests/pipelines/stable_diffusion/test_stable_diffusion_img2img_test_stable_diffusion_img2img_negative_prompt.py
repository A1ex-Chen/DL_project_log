def test_stable_diffusion_img2img_negative_prompt(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionImg2ImgPipeline(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    negative_prompt = 'french fries'
    output = sd_pipe(**inputs, negative_prompt=negative_prompt)
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 32, 32, 3)
    expected_slice = np.array([0.4593, 0.3408, 0.4232, 0.4749, 0.4476, 
        0.4115, 0.4357, 0.4733, 0.4663])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
