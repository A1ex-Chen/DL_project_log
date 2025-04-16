def __init__(self, func, y0, f_options, step_size=None, grid_constructor=
    None, **unused_kwargs):
    unused_kwargs.pop('rtol', None)
    unused_kwargs.pop('atol', None)
    _handle_unused_kwargs(self, unused_kwargs)
    del unused_kwargs
    self.func = func
    self.y0 = y0
    self.f_options = f_options
    if step_size is not None and grid_constructor is None:
        self.grid_constructor = self._grid_constructor_from_step_size(step_size
            )
    elif grid_constructor is None:
        self.grid_constructor = lambda f, y0, t: t
    else:
        raise ValueError(
            'step_size and grid_constructor are exclusive arguments.')
