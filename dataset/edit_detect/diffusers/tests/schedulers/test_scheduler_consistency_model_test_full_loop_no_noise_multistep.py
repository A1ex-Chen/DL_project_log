def test_full_loop_no_noise_multistep(self):
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config()
    scheduler = scheduler_class(**scheduler_config)
    timesteps = [106, 0]
    scheduler.set_timesteps(timesteps=timesteps)
    timesteps = scheduler.timesteps
    generator = torch.manual_seed(0)
    model = self.dummy_model()
    sample = self.dummy_sample_deter * scheduler.init_noise_sigma
    for t in timesteps:
        scaled_sample = scheduler.scale_model_input(sample, t)
        residual = model(scaled_sample, t)
        pred_prev_sample = scheduler.step(residual, t, sample, generator=
            generator).prev_sample
        sample = pred_prev_sample
    result_sum = torch.sum(torch.abs(sample))
    result_mean = torch.mean(torch.abs(sample))
    assert abs(result_sum.item() - 347.6357) < 0.01
    assert abs(result_mean.item() - 0.4527) < 0.001
