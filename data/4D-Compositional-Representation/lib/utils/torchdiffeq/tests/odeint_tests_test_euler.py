def test_euler(self):
    f, y0, t_points, sol = problems.construct_problem(TEST_DEVICE, reverse=True
        )
    y = torchdiffeq.odeint(f, y0, t_points, method='euler')
    self.assertLess(rel_error(sol, y), error_tol)
