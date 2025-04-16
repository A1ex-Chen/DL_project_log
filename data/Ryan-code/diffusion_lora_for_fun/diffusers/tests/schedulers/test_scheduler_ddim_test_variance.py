def test_variance(self):
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config()
    scheduler = scheduler_class(**scheduler_config)
    assert torch.sum(torch.abs(scheduler._get_variance(0, 0) - 0.0)) < 1e-05
    assert torch.sum(torch.abs(scheduler._get_variance(420, 400) - 0.14771)
        ) < 1e-05
    assert torch.sum(torch.abs(scheduler._get_variance(980, 960) - 0.3246)
        ) < 1e-05
    assert torch.sum(torch.abs(scheduler._get_variance(0, 0) - 0.0)) < 1e-05
    assert torch.sum(torch.abs(scheduler._get_variance(487, 486) - 0.00979)
        ) < 1e-05
    assert torch.sum(torch.abs(scheduler._get_variance(999, 998) - 0.02)
        ) < 1e-05
