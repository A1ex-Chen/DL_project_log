def test_full_uneven_loop(self):
    scheduler = DPMSolverSinglestepScheduler(**self.get_scheduler_config())
    num_inference_steps = 50
    model = self.dummy_model()
    sample = self.dummy_sample_deter
    scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(scheduler.timesteps[3:]):
        residual = model(sample, t)
        sample = scheduler.step(residual, t, sample).prev_sample
    result_mean = torch.mean(torch.abs(sample))
    assert abs(result_mean.item() - 0.2574) < 0.001
