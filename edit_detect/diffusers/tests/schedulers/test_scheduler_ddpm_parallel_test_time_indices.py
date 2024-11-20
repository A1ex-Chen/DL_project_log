def test_time_indices(self):
    for t in [0, 500, 999]:
        self.check_over_forward(time_step=t)
