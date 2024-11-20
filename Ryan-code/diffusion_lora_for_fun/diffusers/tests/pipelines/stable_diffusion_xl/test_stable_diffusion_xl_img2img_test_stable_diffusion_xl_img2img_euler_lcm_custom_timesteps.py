def test_stable_diffusion_xl_img2img_euler_lcm_custom_timesteps(self):
    device = 'cpu'
    components = self.get_dummy_components(time_cond_proj_dim=256)
    sd_pipe = StableDiffusionXLImg2ImgPipeline(**components)
    sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.config)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    del inputs['num_inference_steps']
    inputs['timesteps'] = [999, 499]
    image = sd_pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 32, 32, 3)
    expected_slice = np.array([0.5604, 0.4352, 0.4717, 0.5844, 0.5101, 
        0.6704, 0.629, 0.546, 0.5286])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
