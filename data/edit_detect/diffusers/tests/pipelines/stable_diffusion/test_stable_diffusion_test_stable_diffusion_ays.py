def test_stable_diffusion_ays(self):
    from diffusers.schedulers import AysSchedules
    timestep_schedule = AysSchedules['StableDiffusionTimesteps']
    sigma_schedule = AysSchedules['StableDiffusionSigmas']
    device = 'cpu'
    components = self.get_dummy_components(time_cond_proj_dim=256)
    sd_pipe = StableDiffusionPipeline(**components)
    sd_pipe.scheduler = EulerDiscreteScheduler.from_config(sd_pipe.
        scheduler.config)
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    inputs['num_inference_steps'] = 10
    output = sd_pipe(**inputs).images
    inputs = self.get_dummy_inputs(device)
    inputs['num_inference_steps'] = None
    inputs['timesteps'] = timestep_schedule
    output_ts = sd_pipe(**inputs).images
    inputs = self.get_dummy_inputs(device)
    inputs['num_inference_steps'] = None
    inputs['sigmas'] = sigma_schedule
    output_sigmas = sd_pipe(**inputs).images
    assert np.abs(output_sigmas.flatten() - output_ts.flatten()).max(
        ) < 0.001, 'ays timesteps and ays sigmas should have the same outputs'
    assert np.abs(output.flatten() - output_ts.flatten()).max(
        ) > 0.001, 'use ays timesteps should have different outputs'
    assert np.abs(output.flatten() - output_sigmas.flatten()).max(
        ) > 0.001, 'use ays sigmas should have different outputs'
