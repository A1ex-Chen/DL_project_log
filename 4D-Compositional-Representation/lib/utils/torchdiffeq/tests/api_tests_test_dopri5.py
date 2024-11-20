def test_dopri5(self):
    f, y0, t_points, sol = construct_problem(TEST_DEVICE)
    tuple_f = lambda t, y: (f(t, y[0]), f(t, y[1]))
    tuple_y0 = y0, y0
    tuple_y = torchdiffeq.odeint(tuple_f, tuple_y0, t_points, method='dopri5')
    max_error0 = (sol - tuple_y[0]).max()
    max_error1 = (sol - tuple_y[1]).max()
    self.assertLess(max_error0, eps)
    self.assertLess(max_error1, eps)
