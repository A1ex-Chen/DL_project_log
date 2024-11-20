def check_over_forward(self, time_step=0, **forward_kwargs):
    kwargs = dict(self.forward_default_kwargs)
    num_inference_steps = kwargs.pop('num_inference_steps', None)
    sample = self.dummy_sample
    residual = 0.1 * sample
    dummy_past_residuals = [residual + 0.2, residual + 0.15, residual + 0.1,
        residual + 0.05]
    for scheduler_class in self.scheduler_classes:
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)
        scheduler.set_timesteps(num_inference_steps)
        scheduler.ets = dummy_past_residuals[:]
        with tempfile.TemporaryDirectory() as tmpdirname:
            scheduler.save_config(tmpdirname)
            new_scheduler = scheduler_class.from_pretrained(tmpdirname)
            new_scheduler.set_timesteps(num_inference_steps)
            new_scheduler.ets = dummy_past_residuals[:]
        output = scheduler.step_prk(residual, time_step, sample, **kwargs
            ).prev_sample
        new_output = new_scheduler.step_prk(residual, time_step, sample, **
            kwargs).prev_sample
        assert torch.sum(torch.abs(output - new_output)
            ) < 1e-05, 'Scheduler outputs are not identical'
        output = scheduler.step_plms(residual, time_step, sample, **kwargs
            ).prev_sample
        new_output = new_scheduler.step_plms(residual, time_step, sample,
            **kwargs).prev_sample
        assert torch.sum(torch.abs(output - new_output)
            ) < 1e-05, 'Scheduler outputs are not identical'
