def test_sigmas(self):
    for sigma_min, sigma_max in zip([0.0001, 0.001, 0.01], [1, 100, 1000]):
        self.check_over_configs(sigma_min=sigma_min, sigma_max=sigma_max)
