def test_variance_type(self):
    for variance in ['fixed_small_log', 'learned_range']:
        self.check_over_configs(variance_type=variance)
