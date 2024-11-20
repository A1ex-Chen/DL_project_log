def test_stable_diffusion_xl_inpaint_euler_lcm(self):
    device = 'cpu'
    components = self.get_dummy_components(time_cond_proj_dim=256)
    sd_pipe = StableDiffusionXLInpaintPipeline(**components)
    sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.config)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = sd_pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 64, 64, 3)
    expected_slice = np.array([0.6611, 0.5569, 0.5531, 0.5471, 0.5918, 
        0.6393, 0.5074, 0.5468, 0.5185])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
