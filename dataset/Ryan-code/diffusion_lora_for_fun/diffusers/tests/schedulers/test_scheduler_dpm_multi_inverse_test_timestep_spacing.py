def test_timestep_spacing(self):
    for timestep_spacing in ['trailing', 'leading']:
        self.check_over_configs(timestep_spacing=timestep_spacing)
