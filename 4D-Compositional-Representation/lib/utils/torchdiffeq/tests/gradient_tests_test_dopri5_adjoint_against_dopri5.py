def test_dopri5_adjoint_against_dopri5(self):
    func, y0, t_points = self.problem()
    ys = torchdiffeq.odeint_adjoint(func, y0, t_points, method='dopri5')
    gradys = torch.rand_like(ys) * 0.1
    ys.backward(gradys)
    adj_y0_grad = y0.grad
    adj_t_grad = t_points.grad
    adj_A_grad = func.A.grad
    self.assertEqual(max_abs(func.unused_module.weight.grad), 0)
    self.assertEqual(max_abs(func.unused_module.bias.grad), 0)
    func, y0, t_points = self.problem()
    ys = torchdiffeq.odeint(func, y0, t_points, method='dopri5')
    ys.backward(gradys)
    self.assertLess(max_abs(y0.grad - adj_y0_grad), 0.0003)
    self.assertLess(max_abs(t_points.grad - adj_t_grad), 0.0001)
    self.assertLess(max_abs(func.A.grad - adj_A_grad), 0.002)
