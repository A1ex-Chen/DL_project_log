def test_variance_learned_range(self):
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config(variance_type='learned_range')
    scheduler = scheduler_class(**scheduler_config)
    predicted_variance = 0.5
    assert scheduler._get_variance(1, predicted_variance=predicted_variance
        ) - -10.171279 < 1e-05
    assert scheduler._get_variance(487, predicted_variance=predicted_variance
        ) - -5.7998052 < 1e-05
    assert scheduler._get_variance(999, predicted_variance=predicted_variance
        ) - -0.0010011 < 1e-05
