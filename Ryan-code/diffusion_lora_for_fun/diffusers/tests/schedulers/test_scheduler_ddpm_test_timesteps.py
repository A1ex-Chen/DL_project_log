def test_timesteps(self):
    for timesteps in [1, 5, 100, 1000]:
        self.check_over_configs(num_train_timesteps=timesteps)
