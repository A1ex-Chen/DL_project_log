def test_variance_fixed_small_log(self):
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config(variance_type=
        'fixed_small_log')
    scheduler = scheduler_class(**scheduler_config)
    assert torch.sum(torch.abs(scheduler._get_variance(0) - 1e-10)) < 1e-05
    assert torch.sum(torch.abs(scheduler._get_variance(487) - 0.0549625)
        ) < 1e-05
    assert torch.sum(torch.abs(scheduler._get_variance(999) - 0.9994987)
        ) < 1e-05
