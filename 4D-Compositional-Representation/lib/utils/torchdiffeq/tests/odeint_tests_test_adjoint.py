def test_adjoint(self):
    for ode in problems.PROBLEMS.keys():
        f, y0, t_points, sol = problems.construct_problem(TEST_DEVICE,
            reverse=True)
        y = torchdiffeq.odeint_adjoint(f, y0, t_points, method='dopri5')
        with self.subTest(ode=ode):
            self.assertLess(rel_error(sol, y), error_tol)
