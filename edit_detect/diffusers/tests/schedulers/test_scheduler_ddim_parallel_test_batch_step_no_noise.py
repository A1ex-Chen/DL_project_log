def test_batch_step_no_noise(self):
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config()
    scheduler = scheduler_class(**scheduler_config)
    num_inference_steps, eta = 10, 0.0
    scheduler.set_timesteps(num_inference_steps)
    model = self.dummy_model()
    sample1 = self.dummy_sample_deter
    sample2 = self.dummy_sample_deter + 0.1
    sample3 = self.dummy_sample_deter - 0.1
    per_sample_batch = sample1.shape[0]
    samples = torch.stack([sample1, sample2, sample3], dim=0)
    timesteps = torch.arange(num_inference_steps)[0:3, None].repeat(1,
        per_sample_batch)
    residual = model(samples.flatten(0, 1), timesteps.flatten(0, 1))
    pred_prev_sample = scheduler.batch_step_no_noise(residual, timesteps.
        flatten(0, 1), samples.flatten(0, 1), eta)
    result_sum = torch.sum(torch.abs(pred_prev_sample))
    result_mean = torch.mean(torch.abs(pred_prev_sample))
    assert abs(result_sum.item() - 1147.7904) < 0.01
    assert abs(result_mean.item() - 0.4982) < 0.001
