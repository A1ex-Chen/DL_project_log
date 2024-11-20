def test_from_save_pretrained(self):
    kwargs = dict(self.forward_default_kwargs)
    num_inference_steps = kwargs.pop('num_inference_steps', None)
    for scheduler_class in self.scheduler_classes:
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)
        sample = self.dummy_sample
        residual = 0.1 * sample
        with tempfile.TemporaryDirectory() as tmpdirname:
            scheduler.save_config(tmpdirname)
            new_scheduler = scheduler_class.from_pretrained(tmpdirname)
        scheduler.set_timesteps(num_inference_steps)
        new_scheduler.set_timesteps(num_inference_steps)
        timestep = scheduler.timesteps[0]
        sample = self.dummy_sample
        scaled_sample = scheduler.scale_model_input(sample, timestep)
        residual = 0.1 * scaled_sample
        new_scaled_sample = new_scheduler.scale_model_input(sample, timestep)
        new_residual = 0.1 * new_scaled_sample
        if 'generator' in set(inspect.signature(scheduler.step).parameters.
            keys()):
            kwargs['generator'] = torch.manual_seed(0)
        output = scheduler.step(residual, timestep, sample, **kwargs
            ).prev_sample
        if 'generator' in set(inspect.signature(scheduler.step).parameters.
            keys()):
            kwargs['generator'] = torch.manual_seed(0)
        new_output = new_scheduler.step(new_residual, timestep, sample, **
            kwargs).prev_sample
        assert torch.sum(torch.abs(output - new_output)
            ) < 1e-05, 'Scheduler outputs are not identical'
