def run_rnn_test(self, rnn, layers, bidir, state_tuple=False):
    for typ in [torch.float, torch.half]:
        x = torch.randn((self.t, self.b, self.h), dtype=typ).requires_grad_()
        hidden_fn = lambda : torch.zeros((layers + layers * bidir, self.b,
            self.h), dtype=typ)
        if state_tuple:
            hidden = hidden_fn(), hidden_fn()
        else:
            hidden = hidden_fn()
        output, _ = rnn(x, hidden)
        self.assertEqual(output.type(), HALF)
        output[-1, :, :].float().sum().backward()
        self.assertEqual(x.grad.dtype, x.dtype)
