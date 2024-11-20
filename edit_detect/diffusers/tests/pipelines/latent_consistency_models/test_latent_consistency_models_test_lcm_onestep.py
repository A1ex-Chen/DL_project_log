def test_lcm_onestep(self):
    pipe = LatentConsistencyModelPipeline.from_pretrained(
        'SimianLuo/LCM_Dreamshaper_v7', safety_checker=None)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs(torch_device)
    inputs['num_inference_steps'] = 1
    image = pipe(**inputs).images
    assert image.shape == (1, 512, 512, 3)
    image_slice = image[0, -3:, -3:, -1].flatten()
    expected_slice = np.array([0.1025, 0.0911, 0.0984, 0.0981, 0.0901, 
        0.0918, 0.1055, 0.094, 0.073])
    assert np.abs(image_slice - expected_slice).max() < 0.001
