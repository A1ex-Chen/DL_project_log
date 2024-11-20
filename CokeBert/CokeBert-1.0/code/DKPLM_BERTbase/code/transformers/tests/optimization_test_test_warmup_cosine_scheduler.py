def test_warmup_cosine_scheduler(self):
    scheduler = get_cosine_schedule_with_warmup(self.optimizer,
        num_warmup_steps=2, num_training_steps=10)
    lrs = unwrap_schedule(scheduler, self.num_steps)
    expected_learning_rates = [5.0, 10.0, 9.61, 8.53, 6.91, 5.0, 3.08, 1.46,
        0.38, 0.0]
    self.assertEqual(len(lrs[0]), 1)
    self.assertListAlmostEqual([l[0] for l in lrs], expected_learning_rates,
        tol=0.01)
    scheduler = get_cosine_schedule_with_warmup(self.optimizer,
        num_warmup_steps=2, num_training_steps=10)
    lrs_2 = unwrap_and_save_reload_schedule(scheduler, self.num_steps)
    self.assertListEqual([l[0] for l in lrs], [l[0] for l in lrs_2])
