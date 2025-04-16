def test_time_indices(self):
    for t in [0.1, 0.5, 0.75]:
        self.check_over_forward(time_step=t)
