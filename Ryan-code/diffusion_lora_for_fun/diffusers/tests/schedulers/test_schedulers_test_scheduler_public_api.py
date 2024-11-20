def test_scheduler_public_api(self):
    for scheduler_class in self.scheduler_classes:
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)
        if scheduler_class != VQDiffusionScheduler:
            self.assertTrue(hasattr(scheduler, 'init_noise_sigma'),
                f'{scheduler_class} does not implement a required attribute `init_noise_sigma`'
                )
            self.assertTrue(hasattr(scheduler, 'scale_model_input'),
                f'{scheduler_class} does not implement a required class method `scale_model_input(sample, timestep)`'
                )
        self.assertTrue(hasattr(scheduler, 'step'),
            f'{scheduler_class} does not implement a required class method `step(...)`'
            )
        if scheduler_class != VQDiffusionScheduler:
            sample = self.dummy_sample
            if scheduler_class == CMStochasticIterativeScheduler:
                scaled_sigma_max = scheduler.sigma_to_t(scheduler.config.
                    sigma_max)
                scaled_sample = scheduler.scale_model_input(sample,
                    scaled_sigma_max)
            elif scheduler_class == EDMEulerScheduler:
                scaled_sample = scheduler.scale_model_input(sample,
                    scheduler.timesteps[-1])
            else:
                scaled_sample = scheduler.scale_model_input(sample, 0.0)
            self.assertEqual(sample.shape, scaled_sample.shape)
