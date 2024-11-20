def test_adams(self):
    f, y0, t_points, _ = construct_problem(TEST_DEVICE)
    func = lambda y0, t_points: torchdiffeq.odeint(f, y0, t_points, method=
        'adams')
    self.assertTrue(torch.autograd.gradcheck(func, (y0, t_points)))
