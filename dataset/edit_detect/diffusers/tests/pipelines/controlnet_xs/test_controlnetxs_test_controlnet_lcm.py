def test_controlnet_lcm(self):
    device = 'cpu'
    components = self.get_dummy_components(time_cond_proj_dim=8)
    sd_pipe = StableDiffusionControlNetXSPipeline(**components)
    sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    output = sd_pipe(**inputs)
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 16, 16, 3)
    expected_slice = np.array([0.745, 0.753, 0.767, 0.543, 0.523, 0.502, 
        0.314, 0.521, 0.478])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
