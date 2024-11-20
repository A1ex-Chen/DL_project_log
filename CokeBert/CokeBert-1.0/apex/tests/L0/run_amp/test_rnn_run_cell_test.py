def run_cell_test(self, cell, state_tuple=False):
    shape = self.b, self.h
    for typ in [torch.float, torch.half]:
        xs = [torch.randn(shape, dtype=typ).requires_grad_() for _ in range
            (self.t)]
        hidden_fn = lambda : torch.zeros(shape, dtype=typ)
        if state_tuple:
            hidden = hidden_fn(), hidden_fn()
        else:
            hidden = hidden_fn()
        outputs = []
        for i in range(self.t):
            hidden = cell(xs[i], hidden)
            if state_tuple:
                output = hidden[0]
            else:
                output = hidden
            outputs.append(output)
        for y in outputs:
            self.assertEqual(y.type(), HALF)
        outputs[-1].float().sum().backward()
        for i, x in enumerate(xs):
            self.assertEqual(x.grad.dtype, x.dtype)
