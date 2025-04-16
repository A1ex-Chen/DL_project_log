def test_warmup_linear_scheduler(self):
    scheduler = get_linear_schedule_with_warmup(self.optimizer,
        num_warmup_steps=2, num_training_steps=10)
    lrs = unwrap_schedule(scheduler, self.num_steps)
    expected_learning_rates = [5.0, 10.0, 8.75, 7.5, 6.25, 5.0, 3.75, 2.5, 
        1.25, 0.0]
    self.assertEqual(len(lrs[0]), 1)
    self.assertListEqual([l[0] for l in lrs], expected_learning_rates)
    scheduler = get_linear_schedule_with_warmup(self.optimizer,
        num_warmup_steps=2, num_training_steps=10)
    lrs_2 = unwrap_and_save_reload_schedule(scheduler, self.num_steps)
    self.assertListEqual([l[0] for l in lrs], [l[0] for l in lrs_2])
