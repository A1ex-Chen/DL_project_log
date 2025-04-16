def test_schedules(self):
    for schedule in ['linear', 'scaled_linear']:
        self.check_over_configs(beta_schedule=schedule)
