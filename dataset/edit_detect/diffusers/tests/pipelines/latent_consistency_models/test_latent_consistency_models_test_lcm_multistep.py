def test_lcm_multistep(self):
    pipe = LatentConsistencyModelPipeline.from_pretrained(
        'SimianLuo/LCM_Dreamshaper_v7', safety_checker=None)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs(torch_device)
    image = pipe(**inputs).images
    assert image.shape == (1, 512, 512, 3)
    image_slice = image[0, -3:, -3:, -1].flatten()
    expected_slice = np.array([0.01855, 0.01855, 0.01489, 0.01392, 0.01782,
        0.01465, 0.01831, 0.02539, 0.0])
    assert np.abs(image_slice - expected_slice).max() < 0.001
