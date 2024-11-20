def test_betas(self):
    for beta_start, beta_end in zip([0.0001, 0.001, 0.01, 0.1], [0.002, 
        0.02, 0.2, 2]):
        self.check_over_configs(beta_start=beta_start, beta_end=beta_end)
