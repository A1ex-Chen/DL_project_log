def test_full_loop_no_noise(self):
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config()
    scheduler = scheduler_class(**scheduler_config)
    scheduler.set_timesteps(self.num_inference_steps)
    model = self.dummy_model()
    sample = self.dummy_sample_deter * scheduler.init_noise_sigma
    sample = sample.to(torch_device)
    generator = torch.manual_seed(0)
    for i, t in enumerate(scheduler.timesteps):
        sample = scheduler.scale_model_input(sample, t, generator=generator)
        model_output = model(sample, t)
        output = scheduler.step(model_output, t, sample)
        sample = output.prev_sample
    result_sum = torch.sum(torch.abs(sample))
    result_mean = torch.mean(torch.abs(sample))
    if torch_device in ['cpu']:
        assert abs(result_sum.item() - 337.394287109375) < 0.01
        assert abs(result_mean.item() - 0.43931546807289124) < 0.001
    elif torch_device in ['cuda']:
        assert abs(result_sum.item() - 329.1999816894531) < 0.01
        assert abs(result_mean.item() - 0.4286458194255829) < 0.001
    else:
        print('None')
