def test_lambda_min_clipped(self):
    self.check_over_configs(lambda_min_clipped=-float('inf'))
    self.check_over_configs(lambda_min_clipped=-5.1)
