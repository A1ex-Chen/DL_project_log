def test_full_loop_with_v_prediction(self):
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config(prediction_type='v_prediction'
        )
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
        assert abs(result_sum.item() - 124.77149200439453) < 0.01
        assert abs(result_mean.item() - 0.16226289014816284) < 0.001
    elif torch_device in ['cuda']:
        assert abs(result_sum.item() - 128.1663360595703) < 0.01
        assert abs(result_mean.item() - 0.16688326001167297) < 0.001
    else:
        assert abs(result_sum.item() - 119.8487548828125) < 0.01
        assert abs(result_mean.item() - 0.1560530662536621) < 0.001
