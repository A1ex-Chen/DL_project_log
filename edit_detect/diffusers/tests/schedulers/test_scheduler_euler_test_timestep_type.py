def test_timestep_type(self):
    timestep_types = ['discrete', 'continuous']
    for timestep_type in timestep_types:
        self.check_over_configs(timestep_type=timestep_type)
