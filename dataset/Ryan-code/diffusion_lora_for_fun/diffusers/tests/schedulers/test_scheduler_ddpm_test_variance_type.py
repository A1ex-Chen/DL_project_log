def test_variance_type(self):
    for variance in ['fixed_small', 'fixed_large', 'other']:
        self.check_over_configs(variance_type=variance)
