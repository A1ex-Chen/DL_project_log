def test_karras_sigmas(self):
    self.check_over_configs(use_karras_sigmas=True, sigma_min=0.02,
        sigma_max=700.0)
