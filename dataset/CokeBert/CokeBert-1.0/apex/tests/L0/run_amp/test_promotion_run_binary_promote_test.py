def run_binary_promote_test(self, fns, input_shape, x_inplace=False):
    type_pairs = it.product(DTYPES, DTYPES)
    for fn, (xtype, ytype) in it.product(fns, type_pairs):
        x = torch.randn(input_shape, dtype=xtype).requires_grad_()
        x_leaf = x
        if x_inplace:
            x = x.clone()
        y = torch.randn(input_shape, dtype=ytype)
        out = fn(x, y)
        if x_inplace:
            self.assertEqual(out.type(), x.type())
        elif xtype == torch.float or ytype == torch.float:
            self.assertEqual(out.type(), FLOAT)
        else:
            self.assertEqual(out.type(), HALF)
        out.float().sum().backward()
        self.assertEqual(x_leaf.grad.dtype, xtype)
