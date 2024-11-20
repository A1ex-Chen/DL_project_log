def check_over_configs(self, time_step=0, **config):
    kwargs = dict(self.forward_default_kwargs)
    num_inference_steps = kwargs.pop('num_inference_steps', None)
    time_step = time_step if time_step is not None else self.default_timestep
    for scheduler_class in self.scheduler_classes:
        if scheduler_class in (EulerAncestralDiscreteScheduler,
            EulerDiscreteScheduler, LMSDiscreteScheduler):
            time_step = float(time_step)
        scheduler_config = self.get_scheduler_config(**config)
        scheduler = scheduler_class(**scheduler_config)
        if scheduler_class == CMStochasticIterativeScheduler:
            scaled_sigma_max = scheduler.sigma_to_t(scheduler.config.sigma_max)
            time_step = scaled_sigma_max
        if scheduler_class == EDMEulerScheduler:
            time_step = scheduler.timesteps[-1]
        if scheduler_class == VQDiffusionScheduler:
            num_vec_classes = scheduler_config['num_vec_classes']
            sample = self.dummy_sample(num_vec_classes)
            model = self.dummy_model(num_vec_classes)
            residual = model(sample, time_step)
        else:
            sample = self.dummy_sample
            residual = 0.1 * sample
        with tempfile.TemporaryDirectory() as tmpdirname:
            scheduler.save_config(tmpdirname)
            new_scheduler = scheduler_class.from_pretrained(tmpdirname)
        if num_inference_steps is not None and hasattr(scheduler,
            'set_timesteps'):
            scheduler.set_timesteps(num_inference_steps)
            new_scheduler.set_timesteps(num_inference_steps)
        elif num_inference_steps is not None and not hasattr(scheduler,
            'set_timesteps'):
            kwargs['num_inference_steps'] = num_inference_steps
        if scheduler_class == CMStochasticIterativeScheduler:
            _ = scheduler.scale_model_input(sample, scaled_sigma_max)
            _ = new_scheduler.scale_model_input(sample, scaled_sigma_max)
        elif scheduler_class != VQDiffusionScheduler:
            _ = scheduler.scale_model_input(sample, scheduler.timesteps[-1])
            _ = new_scheduler.scale_model_input(sample, scheduler.timesteps[-1]
                )
        if 'generator' in set(inspect.signature(scheduler.step).parameters.
            keys()):
            kwargs['generator'] = torch.manual_seed(0)
        output = scheduler.step(residual, time_step, sample, **kwargs
            ).prev_sample
        if 'generator' in set(inspect.signature(scheduler.step).parameters.
            keys()):
            kwargs['generator'] = torch.manual_seed(0)
        new_output = new_scheduler.step(residual, time_step, sample, **kwargs
            ).prev_sample
        assert torch.sum(torch.abs(output - new_output)
            ) < 1e-05, 'Scheduler outputs are not identical'
