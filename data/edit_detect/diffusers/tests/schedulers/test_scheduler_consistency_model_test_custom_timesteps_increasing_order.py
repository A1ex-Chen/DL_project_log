def test_custom_timesteps_increasing_order(self):
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config()
    scheduler = scheduler_class(**scheduler_config)
    timesteps = [39, 30, 12, 15, 0]
    with self.assertRaises(ValueError, msg=
        '`timesteps` must be in descending order.'):
        scheduler.set_timesteps(timesteps=timesteps)
