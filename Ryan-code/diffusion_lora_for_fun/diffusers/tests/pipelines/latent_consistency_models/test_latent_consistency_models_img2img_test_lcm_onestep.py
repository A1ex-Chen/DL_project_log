def test_lcm_onestep(self):
    pipe = LatentConsistencyModelImg2ImgPipeline.from_pretrained(
        'SimianLuo/LCM_Dreamshaper_v7', safety_checker=None)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs(torch_device)
    inputs['num_inference_steps'] = 1
    image = pipe(**inputs).images
    assert image.shape == (1, 512, 512, 3)
    image_slice = image[0, -3:, -3:, -1].flatten()
    expected_slice = np.array([0.195, 0.1961, 0.2308, 0.1786, 0.1837, 0.232,
        0.1898, 0.1885, 0.2309])
    assert np.abs(image_slice - expected_slice).max() < 0.001
