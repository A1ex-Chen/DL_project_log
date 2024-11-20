def test_stable_diffusion_img2img_default_case_lcm(self):
    device = 'cpu'
    components = self.get_dummy_components(time_cond_proj_dim=256)
    sd_pipe = StableDiffusionImg2ImgPipeline(**components)
    sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = sd_pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 32, 32, 3)
    expected_slice = np.array([0.5709, 0.4614, 0.4587, 0.5978, 0.5298, 
        0.691, 0.624, 0.5212, 0.5454])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
