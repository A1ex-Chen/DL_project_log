def test_full_loop_no_noise_onestep(self):
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config()
    scheduler = scheduler_class(**scheduler_config)
    num_inference_steps = 1
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps
    generator = torch.manual_seed(0)
    model = self.dummy_model()
    sample = self.dummy_sample_deter * scheduler.init_noise_sigma
    for i, t in enumerate(timesteps):
        scaled_sample = scheduler.scale_model_input(sample, t)
        residual = model(scaled_sample, t)
        pred_prev_sample = scheduler.step(residual, t, sample, generator=
            generator).prev_sample
        sample = pred_prev_sample
    result_sum = torch.sum(torch.abs(sample))
    result_mean = torch.mean(torch.abs(sample))
    assert abs(result_sum.item() - 192.7614) < 0.01
    assert abs(result_mean.item() - 0.251) < 0.001
