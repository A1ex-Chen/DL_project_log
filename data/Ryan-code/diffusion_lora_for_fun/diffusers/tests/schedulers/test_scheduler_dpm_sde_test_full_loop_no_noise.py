def test_full_loop_no_noise(self):
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config()
    scheduler = scheduler_class(**scheduler_config)
    scheduler.set_timesteps(self.num_inference_steps)
    model = self.dummy_model()
    sample = self.dummy_sample_deter * scheduler.init_noise_sigma
    sample = sample.to(torch_device)
    for i, t in enumerate(scheduler.timesteps):
        sample = scheduler.scale_model_input(sample, t)
        model_output = model(sample, t)
        output = scheduler.step(model_output, t, sample)
        sample = output.prev_sample
    result_sum = torch.sum(torch.abs(sample))
    result_mean = torch.mean(torch.abs(sample))
    if torch_device in ['mps']:
        assert abs(result_sum.item() - 167.47821044921875) < 0.01
        assert abs(result_mean.item() - 0.2178705964565277) < 0.001
    elif torch_device in ['cuda']:
        assert abs(result_sum.item() - 171.59352111816406) < 0.01
        assert abs(result_mean.item() - 0.22342906892299652) < 0.001
    else:
        assert abs(result_sum.item() - 162.52383422851562) < 0.01
        assert abs(result_mean.item() - 0.211619570851326) < 0.001
