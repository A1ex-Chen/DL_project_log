def test_custom_timesteps_too_large(self):
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config()
    scheduler = scheduler_class(**scheduler_config)
    timesteps = [scheduler.config.num_train_timesteps]
    with self.assertRaises(ValueError, msg=
        '`timesteps` must start before `self.config.train_timesteps`: {scheduler.config.num_train_timesteps}}'
        ):
        scheduler.set_timesteps(timesteps=timesteps)
