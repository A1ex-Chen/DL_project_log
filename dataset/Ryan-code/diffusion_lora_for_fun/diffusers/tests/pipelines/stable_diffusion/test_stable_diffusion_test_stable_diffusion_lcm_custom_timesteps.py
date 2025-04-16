def test_stable_diffusion_lcm_custom_timesteps(self):
    device = 'cpu'
    components = self.get_dummy_components(time_cond_proj_dim=256)
    sd_pipe = StableDiffusionPipeline(**components)
    sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    del inputs['num_inference_steps']
    inputs['timesteps'] = [999, 499]
    output = sd_pipe(**inputs)
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 64, 64, 3)
    expected_slice = np.array([0.2368, 0.49, 0.5019, 0.2723, 0.4473, 0.4578,
        0.4551, 0.3532, 0.4133])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
