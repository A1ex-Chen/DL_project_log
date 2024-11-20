def test_time_indices(self):
    for time_step in [0, 500, 999]:
        for prev_timestep in [None, 5, 100, 250, 500, 750]:
            if prev_timestep is not None and prev_timestep >= time_step:
                continue
            self.check_over_forward(time_step=time_step, prev_timestep=
                prev_timestep)
