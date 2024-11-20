def test_full_loop_with_v_prediction(self):
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config(prediction_type='v_prediction'
        )
    scheduler = scheduler_class(**scheduler_config)
    scheduler.set_timesteps(self.num_inference_steps)
    generator = torch.manual_seed(0)
    model = self.dummy_model()
    sample = self.dummy_sample_deter * scheduler.init_noise_sigma
    sample = sample.to(torch_device)
    for i, t in enumerate(scheduler.timesteps):
        sample = scheduler.scale_model_input(sample, t)
        model_output = model(sample, t)
        output = scheduler.step(model_output, t, sample, generator=generator)
        sample = output.prev_sample
    result_sum = torch.sum(torch.abs(sample))
    result_mean = torch.mean(torch.abs(sample))
    assert abs(result_sum.item() - 108.4439) < 0.01
    assert abs(result_mean.item() - 0.1412) < 0.001
