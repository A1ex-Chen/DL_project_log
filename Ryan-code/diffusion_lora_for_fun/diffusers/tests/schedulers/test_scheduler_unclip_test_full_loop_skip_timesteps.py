def test_full_loop_skip_timesteps(self):
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config()
    scheduler = scheduler_class(**scheduler_config)
    scheduler.set_timesteps(25)
    timesteps = scheduler.timesteps
    model = self.dummy_model()
    sample = self.dummy_sample_deter
    generator = torch.manual_seed(0)
    for i, t in enumerate(timesteps):
        residual = model(sample, t)
        if i + 1 == timesteps.shape[0]:
            prev_timestep = None
        else:
            prev_timestep = timesteps[i + 1]
        pred_prev_sample = scheduler.step(residual, t, sample,
            prev_timestep=prev_timestep, generator=generator).prev_sample
        sample = pred_prev_sample
    result_sum = torch.sum(torch.abs(sample))
    result_mean = torch.mean(torch.abs(sample))
    assert abs(result_sum.item() - 258.2044983) < 0.01
    assert abs(result_mean.item() - 0.3362038) < 0.001
