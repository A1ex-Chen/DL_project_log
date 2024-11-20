def test_timesteps(self):
    for timesteps in [100, 500, 1000]:
        self.check_over_configs(time_step=timesteps - 1,
            num_train_timesteps=timesteps)
