def test_adjoint(self):
    """
        Test against dopri5
        """
    f, y0, t_points, _ = construct_problem(TEST_DEVICE)
    func = lambda y0, t_points: torchdiffeq.odeint(f, y0, t_points, method=
        'dopri5')
    ys = func(y0, t_points)
    torch.manual_seed(0)
    gradys = torch.rand_like(ys)
    ys.backward(gradys)
    reg_t_grad = t_points.grad
    reg_a_grad = f.a.grad
    reg_b_grad = f.b.grad
    f, y0, t_points, _ = construct_problem(TEST_DEVICE)
    func = lambda y0, t_points: torchdiffeq.odeint_adjoint(f, y0, t_points,
        method='dopri5')
    ys = func(y0, t_points)
    ys.backward(gradys)
    adj_t_grad = t_points.grad
    adj_a_grad = f.a.grad
    adj_b_grad = f.b.grad
    self.assertLess(max_abs(reg_t_grad - adj_t_grad), eps)
    self.assertLess(max_abs(reg_a_grad - adj_a_grad), eps)
    self.assertLess(max_abs(reg_b_grad - adj_b_grad), eps)
