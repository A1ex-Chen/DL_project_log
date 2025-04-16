def test_warmup_cosine_hard_restart_scheduler(self):
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(self.
        optimizer, num_warmup_steps=2, num_cycles=2, num_training_steps=10)
    lrs = unwrap_schedule(scheduler, self.num_steps)
    expected_learning_rates = [5.0, 10.0, 8.53, 5.0, 1.46, 10.0, 8.53, 5.0,
        1.46, 0.0]
    self.assertEqual(len(lrs[0]), 1)
    self.assertListAlmostEqual([l[0] for l in lrs], expected_learning_rates,
        tol=0.01)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(self.
        optimizer, num_warmup_steps=2, num_cycles=2, num_training_steps=10)
    lrs_2 = unwrap_and_save_reload_schedule(scheduler, self.num_steps)
    self.assertListEqual([l[0] for l in lrs], [l[0] for l in lrs_2])
