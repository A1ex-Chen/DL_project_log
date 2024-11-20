def test_full_loop_device_karras_sigmas(self):
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config()
    scheduler = scheduler_class(**scheduler_config, use_karras_sigmas=True)
    scheduler.set_timesteps(self.num_inference_steps, device=torch_device)
    generator = torch.manual_seed(0)
    model = self.dummy_model()
    sample = self.dummy_sample_deter * scheduler.init_noise_sigma.cpu()
    sample = sample.to(torch_device)
    for t in scheduler.timesteps:
        sample = scheduler.scale_model_input(sample, t)
        model_output = model(sample, t)
        output = scheduler.step(model_output, t, sample, generator=generator)
        sample = output.prev_sample
    result_sum = torch.sum(torch.abs(sample))
    result_mean = torch.mean(torch.abs(sample))
    assert abs(result_sum.item() - 124.52299499511719) < 0.01
    assert abs(result_mean.item() - 0.16213932633399963) < 0.001
