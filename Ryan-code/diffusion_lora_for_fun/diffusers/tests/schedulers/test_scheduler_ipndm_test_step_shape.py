def test_step_shape(self):
    kwargs = dict(self.forward_default_kwargs)
    num_inference_steps = kwargs.pop('num_inference_steps', None)
    for scheduler_class in self.scheduler_classes:
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)
        sample = self.dummy_sample
        residual = 0.1 * sample
        if num_inference_steps is not None and hasattr(scheduler,
            'set_timesteps'):
            scheduler.set_timesteps(num_inference_steps)
        elif num_inference_steps is not None and not hasattr(scheduler,
            'set_timesteps'):
            kwargs['num_inference_steps'] = num_inference_steps
        dummy_past_residuals = [residual + 0.2, residual + 0.15, residual +
            0.1, residual + 0.05]
        scheduler.ets = dummy_past_residuals[:]
        time_step_0 = scheduler.timesteps[5]
        time_step_1 = scheduler.timesteps[6]
        output_0 = scheduler.step(residual, time_step_0, sample, **kwargs
            ).prev_sample
        output_1 = scheduler.step(residual, time_step_1, sample, **kwargs
            ).prev_sample
        self.assertEqual(output_0.shape, sample.shape)
        self.assertEqual(output_0.shape, output_1.shape)
        output_0 = scheduler.step(residual, time_step_0, sample, **kwargs
            ).prev_sample
        output_1 = scheduler.step(residual, time_step_1, sample, **kwargs
            ).prev_sample
        self.assertEqual(output_0.shape, sample.shape)
        self.assertEqual(output_0.shape, output_1.shape)
