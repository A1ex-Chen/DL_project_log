def test_dopri5(self):
    f, y0, t_points, _ = construct_problem(TEST_DEVICE)
    func = lambda y0, t_points: torchdiffeq.odeint(f, y0, t_points, method=
        'dopri5')
    self.assertTrue(torch.autograd.gradcheck(func, (y0, t_points)))
