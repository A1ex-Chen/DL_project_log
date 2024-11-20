def test_step_shape(self):
    kwargs = dict(self.forward_default_kwargs)
    num_inference_steps = kwargs.pop('num_inference_steps', None)
    for scheduler_class in self.scheduler_classes:
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)
        state = scheduler.create_state()
        sample, _ = self.dummy_sample
        residual = 0.1 * sample
        if num_inference_steps is not None and hasattr(scheduler,
            'set_timesteps'):
            state = scheduler.set_timesteps(state, num_inference_steps,
                shape=sample.shape)
        elif num_inference_steps is not None and not hasattr(scheduler,
            'set_timesteps'):
            kwargs['num_inference_steps'] = num_inference_steps
        dummy_past_residuals = jnp.array([residual + 0.2, residual + 0.15, 
            residual + 0.1, residual + 0.05])
        state = state.replace(ets=dummy_past_residuals[:])
        output_0, state = scheduler.step_prk(state, residual, 0, sample, **
            kwargs)
        output_1, state = scheduler.step_prk(state, residual, 1, sample, **
            kwargs)
        self.assertEqual(output_0.shape, sample.shape)
        self.assertEqual(output_0.shape, output_1.shape)
        output_0, state = scheduler.step_plms(state, residual, 0, sample,
            **kwargs)
        output_1, state = scheduler.step_plms(state, residual, 1, sample,
            **kwargs)
        self.assertEqual(output_0.shape, sample.shape)
        self.assertEqual(output_0.shape, output_1.shape)
