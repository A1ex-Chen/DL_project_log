def test_stable_diffusion_dpm(self):
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', safety_checker=None)
    sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.
        scheduler.config, final_sigmas_type='sigma_min')
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs(torch_device)
    image = sd_pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1].flatten()
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.03503, 0.03494, 0.01087, 0.03128, 0.02552,
        0.00803, 0.00742, 0.00372, 0.0])
    assert np.abs(image_slice - expected_slice).max() < 0.003
