def test_lcm_multistep(self):
    pipe = LatentConsistencyModelImg2ImgPipeline.from_pretrained(
        'SimianLuo/LCM_Dreamshaper_v7', safety_checker=None)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs(torch_device)
    image = pipe(**inputs).images
    assert image.shape == (1, 512, 512, 3)
    image_slice = image[0, -3:, -3:, -1].flatten()
    expected_slice = np.array([0.3756, 0.3816, 0.3767, 0.3718, 0.3739, 
        0.3735, 0.3863, 0.3803, 0.3563])
    assert np.abs(image_slice - expected_slice).max() < 0.001
