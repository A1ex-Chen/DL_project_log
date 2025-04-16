def test_add_noise_device(self, num_inference_steps=10):
    for scheduler_class in self.scheduler_classes:
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)
        scheduler.set_timesteps(num_inference_steps)
        sample = self.dummy_sample.to(torch_device)
        scaled_sample = scheduler.scale_model_input(sample, 0.0)
        self.assertEqual(sample.shape, scaled_sample.shape)
        noise = torch.randn_like(scaled_sample).to(torch_device)
        t = scheduler.timesteps[5][None]
        noised = scheduler.add_noise(scaled_sample, noise, t)
        self.assertEqual(noised.shape, scaled_sample.shape)
