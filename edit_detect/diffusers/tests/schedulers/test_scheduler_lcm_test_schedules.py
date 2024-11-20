def test_schedules(self):
    for schedule in ['linear', 'scaled_linear', 'squaredcos_cap_v2']:
        self.check_over_configs(time_step=self.default_valid_timestep,
            beta_schedule=schedule)
