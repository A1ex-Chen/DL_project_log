def test_full_loop_device(self):
    if torch_device == 'mps':
        return
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config()
    scheduler = scheduler_class(**scheduler_config)
    scheduler.set_timesteps(self.num_inference_steps, device=torch_device)
    generator = torch.manual_seed(0)
    model = self.dummy_model()
    sample = self.dummy_sample_deter.to(torch_device
        ) * scheduler.init_noise_sigma
    for t in scheduler.timesteps:
        sample = scheduler.scale_model_input(sample, t)
        model_output = model(sample, t)
        output = scheduler.step(model_output, t, sample, generator=generator)
        sample = output.prev_sample
    result_sum = torch.sum(torch.abs(sample))
    result_mean = torch.mean(torch.abs(sample))
    assert abs(result_sum.item() - 13849.3818) < 0.1
    assert abs(result_mean.item() - 18.0331) < 0.001
