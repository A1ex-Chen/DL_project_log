def test_switch(self):
    scheduler = DPMSolverSinglestepScheduler(**self.get_scheduler_config())
    sample = self.full_loop(scheduler=scheduler)
    result_mean = torch.mean(torch.abs(sample))
    assert abs(result_mean.item() - 0.2791) < 0.001
    scheduler = DEISMultistepScheduler.from_config(scheduler.config)
    scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)
    scheduler = UniPCMultistepScheduler.from_config(scheduler.config)
    scheduler = DPMSolverSinglestepScheduler.from_config(scheduler.config)
    sample = self.full_loop(scheduler=scheduler)
    result_mean = torch.mean(torch.abs(sample))
    assert abs(result_mean.item() - 0.2791) < 0.001
