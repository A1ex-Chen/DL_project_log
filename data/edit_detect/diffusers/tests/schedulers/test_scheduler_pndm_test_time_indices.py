def test_time_indices(self):
    for t in [1, 5, 10]:
        self.check_over_forward(time_step=t)
