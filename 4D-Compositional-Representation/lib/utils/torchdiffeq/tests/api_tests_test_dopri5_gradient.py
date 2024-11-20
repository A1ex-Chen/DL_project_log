def test_dopri5_gradient(self):
    f, y0, t_points, sol = construct_problem(TEST_DEVICE)
    tuple_f = lambda t, y: (f(t, y[0]), f(t, y[1]))
    for i in range(2):
        func = lambda y0, t_points: torchdiffeq.odeint(tuple_f, (y0, y0),
            t_points, method='dopri5')[i]
        self.assertTrue(torch.autograd.gradcheck(func, (y0, t_points)))
