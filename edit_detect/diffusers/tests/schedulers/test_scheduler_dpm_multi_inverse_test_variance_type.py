def test_variance_type(self):
    self.check_over_configs(variance_type=None)
    self.check_over_configs(variance_type='learned_range')
