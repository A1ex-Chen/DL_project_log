def test_fp16_support(self):
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config(thresholding=True,
        dynamic_thresholding_ratio=0)
    scheduler = scheduler_class(**scheduler_config)
    num_inference_steps = 10
    model = self.dummy_model()
    sample = self.dummy_sample_deter.half()
    scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(scheduler.timesteps):
        residual = model(sample, t)
        sample = scheduler.step(residual, t, sample).prev_sample
    assert sample.dtype == torch.float16
