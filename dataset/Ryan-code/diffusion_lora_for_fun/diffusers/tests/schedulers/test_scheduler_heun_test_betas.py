def test_betas(self):
    for beta_start, beta_end in zip([1e-05, 0.0001, 0.001], [0.0002, 0.002,
        0.02]):
        self.check_over_configs(beta_start=beta_start, beta_end=beta_end)
