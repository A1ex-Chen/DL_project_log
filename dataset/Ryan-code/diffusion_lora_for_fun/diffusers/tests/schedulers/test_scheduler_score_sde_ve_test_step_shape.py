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
        output_0 = scheduler.step_pred(residual, 0, sample, generator=torch
            .manual_seed(0), **kwargs).prev_sample
        output_1 = scheduler.step_pred(residual, 1, sample, generator=torch
            .manual_seed(0), **kwargs).prev_sample
        self.assertEqual(output_0.shape, sample.shape)
        self.assertEqual(output_0.shape, output_1.shape)
