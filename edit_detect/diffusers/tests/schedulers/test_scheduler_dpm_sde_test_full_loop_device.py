def test_full_loop_device(self):
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config()
    scheduler = scheduler_class(**scheduler_config)
    scheduler.set_timesteps(self.num_inference_steps, device=torch_device)
    model = self.dummy_model()
    sample = self.dummy_sample_deter.to(torch_device
        ) * scheduler.init_noise_sigma
    for t in scheduler.timesteps:
        sample = scheduler.scale_model_input(sample, t)
        model_output = model(sample, t)
        output = scheduler.step(model_output, t, sample)
        sample = output.prev_sample
    result_sum = torch.sum(torch.abs(sample))
    result_mean = torch.mean(torch.abs(sample))
    if torch_device in ['mps']:
        assert abs(result_sum.item() - 167.46957397460938) < 0.01
        assert abs(result_mean.item() - 0.21805934607982635) < 0.001
    elif torch_device in ['cuda']:
        assert abs(result_sum.item() - 171.59353637695312) < 0.01
        assert abs(result_mean.item() - 0.22342908382415771) < 0.001
    else:
        assert abs(result_sum.item() - 162.52383422851562) < 0.01
        assert abs(result_mean.item() - 0.211619570851326) < 0.001