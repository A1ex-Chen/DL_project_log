def test_full_loop_with_noise(self):
    if torch_device == 'mps':
        return
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config()
    scheduler = scheduler_class(**scheduler_config)
    scheduler.set_timesteps(self.num_inference_steps)
    generator = torch.manual_seed(0)
    model = self.dummy_model()
    sample = self.dummy_sample_deter * scheduler.init_noise_sigma
    t_start = self.num_inference_steps - 2
    noise = self.dummy_noise_deter
    noise = noise.to(sample.device)
    timesteps = scheduler.timesteps[t_start * scheduler.order:]
    sample = scheduler.add_noise(sample, noise, timesteps[:1])
    for i, t in enumerate(timesteps):
        sample = scheduler.scale_model_input(sample, t)
        model_output = model(sample, t)
        output = scheduler.step(model_output, t, sample, generator=generator)
        sample = output.prev_sample
    result_sum = torch.sum(torch.abs(sample))
    result_mean = torch.mean(torch.abs(sample))
    assert abs(result_sum.item() - 93087.0312
        ) < 0.01, f' expected result sum 93087.0312, but get {result_sum}'
    assert abs(result_mean.item() - 121.2071
        ) < 0.005, f' expected result mean 121.2071, but get {result_mean}'
