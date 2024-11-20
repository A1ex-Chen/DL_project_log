def test_schedules(self):
    for schedule in ['linear', 'scaled_linear', 'exp']:
        self.check_over_configs(beta_schedule=schedule)
