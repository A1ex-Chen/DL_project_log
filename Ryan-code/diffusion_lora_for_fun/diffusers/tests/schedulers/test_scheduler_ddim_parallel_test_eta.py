def test_eta(self):
    for t, eta in zip([1, 10, 49], [0.0, 0.5, 1.0]):
        self.check_over_forward(time_step=t, eta=eta)
