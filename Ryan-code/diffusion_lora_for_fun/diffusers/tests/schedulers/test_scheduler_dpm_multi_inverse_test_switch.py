def test_switch(self):
    scheduler = DPMSolverMultistepInverseScheduler(**self.
        get_scheduler_config())
    sample = self.full_loop(scheduler=scheduler)
    result_mean = torch.mean(torch.abs(sample))
    assert abs(result_mean.item() - 0.7047) < 0.001
    scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)
    scheduler = DPMSolverMultistepInverseScheduler.from_config(scheduler.config
        )
    sample = self.full_loop(scheduler=scheduler)
    new_result_mean = torch.mean(torch.abs(sample))
    assert abs(new_result_mean.item() - result_mean.item()) < 0.001
