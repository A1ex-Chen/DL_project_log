def test_step_shape(self):
    kwargs = dict(self.forward_default_kwargs)
    num_inference_steps = kwargs.pop('num_inference_steps', None)
    for scheduler_class in self.scheduler_classes:
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)
        sample = self.dummy_sample
        residual = 0.1 * sample
        scheduler.set_timesteps(num_inference_steps)
        timestep_0 = scheduler.timesteps[-2]
        timestep_1 = scheduler.timesteps[-1]
        output_0 = scheduler.step(residual, timestep_0, sample, **kwargs
            ).prev_sample
        output_1 = scheduler.step(residual, timestep_1, sample, **kwargs
            ).prev_sample
        self.assertEqual(output_0.shape, sample.shape)
        self.assertEqual(output_0.shape, output_1.shape)
