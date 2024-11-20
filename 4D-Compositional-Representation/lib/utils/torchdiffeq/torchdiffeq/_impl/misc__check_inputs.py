def _check_inputs(func, y0, t, f_options):
    tensor_input = False
    if torch.is_tensor(y0):
        tensor_input = True
        y0 = y0,
        _base_nontuple_func_ = func
        func = lambda t, y, **f_options: (_base_nontuple_func_(t, y[0], **
            f_options),)
    assert isinstance(y0, tuple), 'y0 must be either a torch.Tensor or a tuple'
    for y0_ in y0:
        assert torch.is_tensor(y0_
            ), 'each element must be a torch.Tensor but received {}'.format(
            type(y0_))
    if _decreasing(t):
        t = -t
        _base_reverse_func = func
        func = lambda t, y, **f_options: tuple(-f_ for f_ in
            _base_reverse_func(-t, y, **f_options))
    for y0_ in y0:
        if not torch.is_floating_point(y0_):
            raise TypeError('`y0` must be a floating point Tensor but is a {}'
                .format(y0_.type()))
    if not torch.is_floating_point(t):
        raise TypeError('`t` must be a floating point Tensor but is a {}'.
            format(t.type()))
    return tensor_input, func, y0, t
