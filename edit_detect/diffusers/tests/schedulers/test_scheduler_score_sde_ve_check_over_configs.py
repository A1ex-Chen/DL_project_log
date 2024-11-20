def check_over_configs(self, time_step=0, **config):
    kwargs = dict(self.forward_default_kwargs)
    for scheduler_class in self.scheduler_classes:
        sample = self.dummy_sample
        residual = 0.1 * sample
        scheduler_config = self.get_scheduler_config(**config)
        scheduler = scheduler_class(**scheduler_config)
        with tempfile.TemporaryDirectory() as tmpdirname:
            scheduler.save_config(tmpdirname)
            new_scheduler = scheduler_class.from_pretrained(tmpdirname)
        output = scheduler.step_pred(residual, time_step, sample, generator
            =torch.manual_seed(0), **kwargs).prev_sample
        new_output = new_scheduler.step_pred(residual, time_step, sample,
            generator=torch.manual_seed(0), **kwargs).prev_sample
        assert torch.sum(torch.abs(output - new_output)
            ) < 1e-05, 'Scheduler outputs are not identical'
        output = scheduler.step_correct(residual, sample, generator=torch.
            manual_seed(0), **kwargs).prev_sample
        new_output = new_scheduler.step_correct(residual, sample, generator
            =torch.manual_seed(0), **kwargs).prev_sample
        assert torch.sum(torch.abs(output - new_output)
            ) < 1e-05, 'Scheduler correction are not identical'
