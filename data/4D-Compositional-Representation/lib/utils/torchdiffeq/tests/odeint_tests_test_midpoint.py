def test_midpoint(self):
    f, y0, t_points, sol = problems.construct_problem(TEST_DEVICE, reverse=True
        )
    y = torchdiffeq.odeint(f, y0, t_points[0:1], method='midpoint')
    self.assertLess(max_abs(sol[0] - y), error_tol)
