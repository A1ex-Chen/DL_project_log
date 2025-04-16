def test_step_shape(self):
    num_inference_steps = 10
    scheduler_config = self.get_scheduler_config()
    scheduler = self.scheduler_classes[0](**scheduler_config)
    scheduler.set_timesteps(num_inference_steps)
    timestep_0 = scheduler.timesteps[0]
    timestep_1 = scheduler.timesteps[1]
    sample = self.dummy_sample
    residual = 0.1 * sample
    output_0 = scheduler.step(residual, timestep_0, sample).prev_sample
    output_1 = scheduler.step(residual, timestep_1, sample).prev_sample
    self.assertEqual(output_0.shape, sample.shape)
    self.assertEqual(output_0.shape, output_1.shape)
