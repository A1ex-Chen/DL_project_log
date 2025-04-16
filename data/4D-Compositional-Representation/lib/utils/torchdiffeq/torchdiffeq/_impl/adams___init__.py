def __init__(self, func, y0, f_options, rtol, atol, implicit=True,
    max_order=_MAX_ORDER, safety=0.9, ifactor=10.0, dfactor=0.2, **
    unused_kwargs):
    _handle_unused_kwargs(self, unused_kwargs)
    del unused_kwargs
    self.func = func
    self.y0 = y0
    self.rtol = rtol if _is_iterable(rtol) else [rtol] * len(y0)
    self.atol = atol if _is_iterable(atol) else [atol] * len(y0)
    self.f_options = f_options
    self.implicit = implicit
    self.max_order = int(max(_MIN_ORDER, min(max_order, _MAX_ORDER)))
    self.safety = _convert_to_tensor(safety, dtype=torch.float64, device=y0
        [0].device)
    self.ifactor = _convert_to_tensor(ifactor, dtype=torch.float64, device=
        y0[0].device)
    self.dfactor = _convert_to_tensor(dfactor, dtype=torch.float64, device=
        y0[0].device)
