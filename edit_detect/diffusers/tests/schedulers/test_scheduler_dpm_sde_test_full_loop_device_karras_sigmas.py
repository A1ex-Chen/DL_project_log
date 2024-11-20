def test_full_loop_device_karras_sigmas(self):
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config()
    scheduler = scheduler_class(**scheduler_config, use_karras_sigmas=True)
    scheduler.set_timesteps(self.num_inference_steps, device=torch_device)
    model = self.dummy_model()
    sample = self.dummy_sample_deter.to(torch_device
        ) * scheduler.init_noise_sigma
    sample = sample.to(torch_device)
    for t in scheduler.timesteps:
        sample = scheduler.scale_model_input(sample, t)
        model_output = model(sample, t)
        output = scheduler.step(model_output, t, sample)
        sample = output.prev_sample
    result_sum = torch.sum(torch.abs(sample))
    result_mean = torch.mean(torch.abs(sample))
    if torch_device in ['mps']:
        assert abs(result_sum.item() - 176.66974135742188) < 0.01
        assert abs(result_mean.item() - 0.23003872730981811) < 0.01
    elif torch_device in ['cuda']:
        assert abs(result_sum.item() - 177.63653564453125) < 0.01
        assert abs(result_mean.item() - 0.23003872730981811) < 0.01
    else:
        assert abs(result_sum.item() - 170.3135223388672) < 0.01
        assert abs(result_mean.item() - 0.23003872730981811) < 0.01
