def test_custom_timesteps_increasing_order(self):
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config()
    scheduler = scheduler_class(**scheduler_config)
    timesteps = [100, 87, 50, 51, 0]
    with self.assertRaises(ValueError, msg=
        '`custom_timesteps` must be in descending order.'):
        scheduler.set_timesteps(timesteps=timesteps)
