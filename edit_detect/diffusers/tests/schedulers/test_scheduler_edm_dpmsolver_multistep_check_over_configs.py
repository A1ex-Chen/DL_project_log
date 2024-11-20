def check_over_configs(self, time_step=0, **config):
    kwargs = dict(self.forward_default_kwargs)
    num_inference_steps = kwargs.pop('num_inference_steps', None)
    sample = self.dummy_sample
    residual = 0.1 * sample
    dummy_past_residuals = [residual + 0.2, residual + 0.15, residual + 0.1]
    for scheduler_class in self.scheduler_classes:
        scheduler_config = self.get_scheduler_config(**config)
        scheduler = scheduler_class(**scheduler_config)
        scheduler.set_timesteps(num_inference_steps)
        scheduler.model_outputs = dummy_past_residuals[:scheduler.config.
            solver_order]
        with tempfile.TemporaryDirectory() as tmpdirname:
            scheduler.save_config(tmpdirname)
            new_scheduler = scheduler_class.from_pretrained(tmpdirname)
            new_scheduler.set_timesteps(num_inference_steps)
            new_scheduler.model_outputs = dummy_past_residuals[:
                new_scheduler.config.solver_order]
        output, new_output = sample, sample
        for t in range(time_step, time_step + scheduler.config.solver_order + 1
            ):
            t = new_scheduler.timesteps[t]
            output = scheduler.step(residual, t, output, **kwargs).prev_sample
            new_output = new_scheduler.step(residual, t, new_output, **kwargs
                ).prev_sample
            assert torch.sum(torch.abs(output - new_output)
                ) < 1e-05, 'Scheduler outputs are not identical'
