def test_stable_diffusion_inpaint_euler(self):
    device = 'cpu'
    components = self.get_dummy_components(time_cond_proj_dim=256)
    sd_pipe = StableDiffusionInpaintPipeline(**components)
    sd_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(sd_pipe
        .scheduler.config)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device, output_pil=False)
    half_dim = inputs['image'].shape[2] // 2
    inputs['mask_image'][0, 0, :half_dim, :half_dim] = 0
    inputs['num_inference_steps'] = 4
    image = sd_pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 64, 64, 3)
    expected_slice = np.array([[0.6387283, 0.5564158, 0.58631873, 0.5539942,
        0.5494673, 0.6461868, 0.5251618, 0.5497595, 0.5508756]])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.0001
