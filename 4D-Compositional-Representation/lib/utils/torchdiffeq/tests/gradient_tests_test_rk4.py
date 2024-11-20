def test_rk4(self):
    f, y0, t_points, _ = construct_problem(TEST_DEVICE)
    func = lambda y0, t_points: torchdiffeq.odeint(f, y0, t_points, method=
        'rk4')
    self.assertTrue(torch.autograd.gradcheck(func, (y0, t_points)))
