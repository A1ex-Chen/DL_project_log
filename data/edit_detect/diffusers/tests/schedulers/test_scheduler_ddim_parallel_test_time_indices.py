def test_time_indices(self):
    for t in [1, 10, 49]:
        self.check_over_forward(time_step=t)
