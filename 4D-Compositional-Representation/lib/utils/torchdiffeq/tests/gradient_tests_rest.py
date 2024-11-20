import unittest
import torch
import torchdiffeq

from problems import construct_problem

eps = 1e-12

torch.set_default_dtype(torch.float64)
TEST_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




class TestGradient(unittest.TestCase):







class TestCompareAdjointGradient(unittest.TestCase):





        y0 = torch.tensor([[2., 0.]]).to(TEST_DEVICE).requires_grad_(True)
        t_points = torch.linspace(0., 25., 10).to(TEST_DEVICE).requires_grad_(True)
        func = Odefunc().to(TEST_DEVICE)
        return func, y0, t_points

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

        self.assertLess(max_abs(y0.grad - adj_y0_grad), 3e-4)
        self.assertLess(max_abs(t_points.grad - adj_t_grad), 1e-4)
        self.assertLess(max_abs(func.A.grad - adj_A_grad), 2e-3)

    def test_adams_adjoint_against_dopri5(self):
        func, y0, t_points = self.problem()
        ys_ = torchdiffeq.odeint_adjoint(func, y0, t_points, method='adams')
        gradys = torch.rand_like(ys_) * 0.1
        ys_.backward(gradys)

        adj_y0_grad = y0.grad
        adj_t_grad = t_points.grad
        adj_A_grad = func.A.grad
        self.assertEqual(max_abs(func.unused_module.weight.grad), 0)
        self.assertEqual(max_abs(func.unused_module.bias.grad), 0)

        func, y0, t_points = self.problem()
        ys = torchdiffeq.odeint(func, y0, t_points, method='dopri5')
        ys.backward(gradys)

        self.assertLess(max_abs(y0.grad - adj_y0_grad), 5e-2)
        self.assertLess(max_abs(t_points.grad - adj_t_grad), 5e-4)
        self.assertLess(max_abs(func.A.grad - adj_A_grad), 2e-2)


if __name__ == '__main__':
    unittest.main()