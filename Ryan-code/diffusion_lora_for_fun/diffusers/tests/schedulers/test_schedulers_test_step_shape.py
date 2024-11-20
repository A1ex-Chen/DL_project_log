def test_step_shape(self):
    kwargs = dict(self.forward_default_kwargs)
    num_inference_steps = kwargs.pop('num_inference_steps', self.
        default_num_inference_steps)
    timestep_0 = self.default_timestep
    timestep_1 = self.default_timestep_2
    for scheduler_class in self.scheduler_classes:
        if scheduler_class in (EulerAncestralDiscreteScheduler,
            EulerDiscreteScheduler, LMSDiscreteScheduler):
            timestep_0 = float(timestep_0)
            timestep_1 = float(timestep_1)
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)
        if scheduler_class == VQDiffusionScheduler:
            num_vec_classes = scheduler_config['num_vec_classes']
            sample = self.dummy_sample(num_vec_classes)
            model = self.dummy_model(num_vec_classes)
            residual = model(sample, timestep_0)
        else:
            sample = self.dummy_sample
            residual = 0.1 * sample
        if num_inference_steps is not None and hasattr(scheduler,
            'set_timesteps'):
            scheduler.set_timesteps(num_inference_steps)
        elif num_inference_steps is not None and not hasattr(scheduler,
            'set_timesteps'):
            kwargs['num_inference_steps'] = num_inference_steps
        output_0 = scheduler.step(residual, timestep_0, sample, **kwargs
            ).prev_sample
        output_1 = scheduler.step(residual, timestep_1, sample, **kwargs
            ).prev_sample
        self.assertEqual(output_0.shape, sample.shape)
        self.assertEqual(output_0.shape, output_1.shape)
