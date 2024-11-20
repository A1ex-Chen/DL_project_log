def test_full_loop_with_noise(self):
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config()
    scheduler = scheduler_class(**scheduler_config)
    num_inference_steps, eta = 10, 0.0
    t_start = 8
    model = self.dummy_model()
    sample = self.dummy_sample_deter
    scheduler.set_timesteps(num_inference_steps)
    noise = self.dummy_noise_deter
    timesteps = scheduler.timesteps[t_start * scheduler.order:]
    sample = scheduler.add_noise(sample, noise, timesteps[:1])
    for t in timesteps:
        residual = model(sample, t)
        sample = scheduler.step(residual, t, sample, eta).prev_sample
    result_sum = torch.sum(torch.abs(sample))
    result_mean = torch.mean(torch.abs(sample))
    assert abs(result_sum.item() - 354.5418
        ) < 0.01, f' expected result sum 218.4379, but get {result_sum}'
    assert abs(result_mean.item() - 0.4616
        ) < 0.001, f' expected result mean 0.2844, but get {result_mean}'
