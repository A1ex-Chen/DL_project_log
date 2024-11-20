def test_add_noise_device(self):
    for scheduler_class in self.scheduler_classes:
        if scheduler_class == IPNDMScheduler:
            continue
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)
        scheduler.set_timesteps(self.default_num_inference_steps)
        sample = self.dummy_sample.to(torch_device)
        if scheduler_class == CMStochasticIterativeScheduler:
            scaled_sigma_max = scheduler.sigma_to_t(scheduler.config.sigma_max)
            scaled_sample = scheduler.scale_model_input(sample,
                scaled_sigma_max)
        elif scheduler_class == EDMEulerScheduler:
            scaled_sample = scheduler.scale_model_input(sample, scheduler.
                timesteps[-1])
        else:
            scaled_sample = scheduler.scale_model_input(sample, 0.0)
        self.assertEqual(sample.shape, scaled_sample.shape)
        noise = torch.randn_like(scaled_sample).to(torch_device)
        t = scheduler.timesteps[5][None]
        noised = scheduler.add_noise(scaled_sample, noise, t)
        self.assertEqual(noised.shape, scaled_sample.shape)
