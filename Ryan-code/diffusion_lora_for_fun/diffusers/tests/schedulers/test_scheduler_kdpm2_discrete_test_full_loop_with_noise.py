def test_full_loop_with_noise(self):
    if torch_device == 'mps':
        return
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config()
    scheduler = scheduler_class(**scheduler_config)
    scheduler.set_timesteps(self.num_inference_steps)
    model = self.dummy_model()
    sample = self.dummy_sample_deter * scheduler.init_noise_sigma
    sample = sample.to(torch_device)
    t_start = self.num_inference_steps - 2
    noise = self.dummy_noise_deter
    noise = noise.to(sample.device)
    timesteps = scheduler.timesteps[t_start * scheduler.order:]
    sample = scheduler.add_noise(sample, noise, timesteps[:1])
    for i, t in enumerate(timesteps):
        sample = scheduler.scale_model_input(sample, t)
        model_output = model(sample, t)
        output = scheduler.step(model_output, t, sample)
        sample = output.prev_sample
    result_sum = torch.sum(torch.abs(sample))
    result_mean = torch.mean(torch.abs(sample))
    assert abs(result_sum.item() - 70408.4062
        ) < 0.01, f' expected result sum 70408.4062, but get {result_sum}'
    assert abs(result_mean.item() - 91.6776
        ) < 0.001, f' expected result mean 91.6776, but get {result_mean}'
