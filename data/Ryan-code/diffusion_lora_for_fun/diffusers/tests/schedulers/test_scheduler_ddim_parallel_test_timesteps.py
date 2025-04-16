def test_timesteps(self):
    for timesteps in [100, 500, 1000]:
        self.check_over_configs(num_train_timesteps=timesteps)
