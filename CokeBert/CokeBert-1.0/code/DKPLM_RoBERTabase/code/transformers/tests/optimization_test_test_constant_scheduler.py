def test_constant_scheduler(self):
    scheduler = get_constant_schedule(self.optimizer)
    lrs = unwrap_schedule(scheduler, self.num_steps)
    expected_learning_rates = [10.0] * self.num_steps
    self.assertEqual(len(lrs[0]), 1)
    self.assertListEqual([l[0] for l in lrs], expected_learning_rates)
    scheduler = get_constant_schedule(self.optimizer)
    lrs_2 = unwrap_and_save_reload_schedule(scheduler, self.num_steps)
    self.assertListEqual([l[0] for l in lrs], [l[0] for l in lrs_2])
