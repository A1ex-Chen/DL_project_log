def test_pow_of_3_inference_steps(self):
    num_inference_steps = 27
    for scheduler_class in self.scheduler_classes:
        sample = self.dummy_sample
        residual = 0.1 * sample
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)
        scheduler.set_timesteps(num_inference_steps)
        for i, t in enumerate(scheduler.prk_timesteps[:2]):
            sample = scheduler.step_prk(residual, t, sample).prev_sample
