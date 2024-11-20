def test_adapter_sdxl_lcm(self):
    device = 'cpu'
    components = self.get_dummy_components(time_cond_proj_dim=256)
    sd_pipe = StableDiffusionXLAdapterPipeline(**components)
    sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    output = sd_pipe(**inputs)
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 64, 64, 3)
    expected_slice = np.array([0.5313, 0.5375, 0.4942, 0.5021, 0.6142, 
        0.4968, 0.5434, 0.5311, 0.5448])
    debug = [str(round(i, 4)) for i in image_slice.flatten().tolist()]
    print(','.join(debug))
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
