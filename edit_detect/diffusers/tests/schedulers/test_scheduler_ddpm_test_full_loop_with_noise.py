def test_full_loop_with_noise(self):
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config()
    scheduler = scheduler_class(**scheduler_config)
    num_trained_timesteps = len(scheduler)
    t_start = num_trained_timesteps - 2
    model = self.dummy_model()
    sample = self.dummy_sample_deter
    generator = torch.manual_seed(0)
    noise = self.dummy_noise_deter
    timesteps = scheduler.timesteps[t_start * scheduler.order:]
    sample = scheduler.add_noise(sample, noise, timesteps[:1])
    for t in timesteps:
        residual = model(sample, t)
        pred_prev_sample = scheduler.step(residual, t, sample, generator=
            generator).prev_sample
        sample = pred_prev_sample
    result_sum = torch.sum(torch.abs(sample))
    result_mean = torch.mean(torch.abs(sample))
    assert abs(result_sum.item() - 387.9466
        ) < 0.01, f' expected result sum 387.9466, but get {result_sum}'
    assert abs(result_mean.item() - 0.5051
        ) < 0.001, f' expected result mean 0.5051, but get {result_mean}'
