def test_stable_diffusion_xl_euler_lcm_custom_timesteps(self):
    device = 'cpu'
    components = self.get_dummy_components(time_cond_proj_dim=256)
    sd_pipe = StableDiffusionXLPipeline(**components)
    sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    del inputs['num_inference_steps']
    inputs['timesteps'] = [999, 499]
    image = sd_pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 64, 64, 3)
    expected_slice = np.array([0.4917, 0.6555, 0.4348, 0.5219, 0.7324, 
        0.4855, 0.5168, 0.5447, 0.5156])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
