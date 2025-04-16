def test_timesteps(self):
    for timesteps in [25, 50, 100, 999, 1000]:
        self.check_over_configs(num_train_timesteps=timesteps)
