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
    if torch_device in ['cpu', 'mps']:
        assert abs(result_sum.item() - 4.6934e-07) < 0.01
        assert abs(result_mean.item() - 6.1112e-10) < 0.001
    else:
        assert abs(result_sum.item() - 4.693428650170972e-07) < 0.01
        assert abs(result_mean.item() - 0.0002) < 0.001
