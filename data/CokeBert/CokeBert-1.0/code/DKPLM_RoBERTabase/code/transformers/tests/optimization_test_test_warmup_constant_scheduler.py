def test_warmup_constant_scheduler(self):
    scheduler = get_constant_schedule_with_warmup(self.optimizer,
        num_warmup_steps=4)
    lrs = unwrap_schedule(scheduler, self.num_steps)
    expected_learning_rates = [2.5, 5.0, 7.5, 10.0, 10.0, 10.0, 10.0, 10.0,
        10.0, 10.0]
    self.assertEqual(len(lrs[0]), 1)
    self.assertListEqual([l[0] for l in lrs], expected_learning_rates)
    scheduler = get_constant_schedule_with_warmup(self.optimizer,
        num_warmup_steps=4)
    lrs_2 = unwrap_and_save_reload_schedule(scheduler, self.num_steps)
    self.assertListEqual([l[0] for l in lrs], [l[0] for l in lrs_2])
