def test_schedules(self):
    for schedule in ['linear', 'squaredcos_cap_v2']:
        self.check_over_configs(beta_schedule=schedule)
