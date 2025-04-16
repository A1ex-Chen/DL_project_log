def test_variance(self):
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config()
    scheduler = scheduler_class(**scheduler_config)
    assert torch.sum(torch.abs(scheduler._get_variance(0) - 0.0)) < 1e-05
    assert torch.sum(torch.abs(scheduler._get_variance(487) - 0.00979)) < 1e-05
    assert torch.sum(torch.abs(scheduler._get_variance(999) - 0.02)) < 1e-05
