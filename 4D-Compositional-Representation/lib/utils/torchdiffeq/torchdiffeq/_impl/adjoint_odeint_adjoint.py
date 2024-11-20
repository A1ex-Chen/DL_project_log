def odeint_adjoint(func, y0, t, rtol=1e-06, atol=1e-12, method=None,
    options=None, f_options=None):
    if not isinstance(func, nn.Module):
        raise ValueError('func is required to be an instance of nn.Module.')
    tensor_input = False
    if torch.is_tensor(y0):


        class TupleFunc(nn.Module):

            def __init__(self, base_func):
                super(TupleFunc, self).__init__()
                self.base_func = base_func

            def forward(self, t, y, **f_options):
                return self.base_func(t, y[0], **f_options),
        tensor_input = True
        y0 = y0,
        func = TupleFunc(func)
    flat_params = _flatten(func.parameters())
    ys = OdeintAdjointMethod.apply(*y0, func, t, flat_params, rtol, atol,
        method, options, f_options)
    if tensor_input:
        ys = ys[0]
    return ys
