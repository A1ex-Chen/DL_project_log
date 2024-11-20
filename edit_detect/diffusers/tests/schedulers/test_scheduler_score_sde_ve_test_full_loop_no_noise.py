def test_full_loop_no_noise(self):
    kwargs = dict(self.forward_default_kwargs)
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config()
    scheduler = scheduler_class(**scheduler_config)
    num_inference_steps = 3
    model = self.dummy_model()
    sample = self.dummy_sample_deter
    scheduler.set_sigmas(num_inference_steps)
    scheduler.set_timesteps(num_inference_steps)
    generator = torch.manual_seed(0)
    for i, t in enumerate(scheduler.timesteps):
        sigma_t = scheduler.sigmas[i]
        for _ in range(scheduler.config.correct_steps):
            with torch.no_grad():
                model_output = model(sample, sigma_t)
            sample = scheduler.step_correct(model_output, sample, generator
                =generator, **kwargs).prev_sample
        with torch.no_grad():
            model_output = model(sample, sigma_t)
        output = scheduler.step_pred(model_output, t, sample, generator=
            generator, **kwargs)
        sample, _ = output.prev_sample, output.prev_sample_mean
    result_sum = torch.sum(torch.abs(sample))
    result_mean = torch.mean(torch.abs(sample))
    assert np.isclose(result_sum.item(), 14372758528.0)
    assert np.isclose(result_mean.item(), 18714530.0)
