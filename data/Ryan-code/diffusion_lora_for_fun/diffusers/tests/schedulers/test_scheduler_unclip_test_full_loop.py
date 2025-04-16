def test_full_loop(self):
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config()
    scheduler = scheduler_class(**scheduler_config)
    timesteps = scheduler.timesteps
    model = self.dummy_model()
    sample = self.dummy_sample_deter
    generator = torch.manual_seed(0)
    for i, t in enumerate(timesteps):
        residual = model(sample, t)
        pred_prev_sample = scheduler.step(residual, t, sample, generator=
            generator).prev_sample
        sample = pred_prev_sample
    result_sum = torch.sum(torch.abs(sample))
    result_mean = torch.mean(torch.abs(sample))
    assert abs(result_sum.item() - 252.2682495) < 0.01
    assert abs(result_mean.item() - 0.3284743) < 0.001
