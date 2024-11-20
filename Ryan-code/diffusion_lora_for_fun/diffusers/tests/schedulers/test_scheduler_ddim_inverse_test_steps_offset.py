def test_steps_offset(self):
    for steps_offset in [0, 1]:
        self.check_over_configs(steps_offset=steps_offset)
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config(steps_offset=1)
    scheduler = scheduler_class(**scheduler_config)
    scheduler.set_timesteps(5)
    assert torch.equal(scheduler.timesteps, torch.LongTensor([1, 201, 401, 
        601, 801]))
