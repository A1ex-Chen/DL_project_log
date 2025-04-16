def test_disabled_linear(self):
    m = nn.Linear(self.h, self.h)
    f = ft.partial(F.linear, weight=m.weight, bias=m.bias)
    input_shape = self.b, self.h
    for fn in [m, f]:
        x = torch.randn(input_shape, dtype=torch.float).requires_grad_()
        y = fn(x)
        self.assertEqual(y.type(), FLOAT)
        y.sum().backward()
        self.assertEqual(x.grad.type(), FLOAT)
        x = torch.randn(input_shape, dtype=torch.half).requires_grad_()
        self.assertRaises(RuntimeError, fn, x)
