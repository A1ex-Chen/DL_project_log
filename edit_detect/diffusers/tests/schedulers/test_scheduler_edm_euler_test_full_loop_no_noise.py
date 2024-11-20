def test_full_loop_no_noise(self, num_inference_steps=10, seed=0):
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config()
    scheduler = scheduler_class(**scheduler_config)
    scheduler.set_timesteps(num_inference_steps)
    model = self.dummy_model()
    sample = self.dummy_sample_deter * scheduler.init_noise_sigma
    for i, t in enumerate(scheduler.timesteps):
        scaled_sample = scheduler.scale_model_input(sample, t)
        model_output = model(scaled_sample, t)
        output = scheduler.step(model_output, t, sample)
        sample = output.prev_sample
    result_sum = torch.sum(torch.abs(sample))
    result_mean = torch.mean(torch.abs(sample))
    assert abs(result_sum.item() - 34.1855) < 0.001
    assert abs(result_mean.item() - 0.044) < 0.001
