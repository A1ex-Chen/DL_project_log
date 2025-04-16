def test_custom_timesteps(self):
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config()
    scheduler = scheduler_class(**scheduler_config)
    timesteps = [100, 87, 50, 1, 0]
    scheduler.set_timesteps(timesteps=timesteps)
    scheduler_timesteps = scheduler.timesteps
    for i, timestep in enumerate(scheduler_timesteps):
        if i == len(timesteps) - 1:
            expected_prev_t = -1
        else:
            expected_prev_t = timesteps[i + 1]
        prev_t = scheduler.previous_timestep(timestep)
        prev_t = prev_t.item()
        self.assertEqual(prev_t, expected_prev_t)
