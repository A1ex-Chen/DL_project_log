def __init__(self, func, y0, rtol, atol, first_step=None, safety=0.9,
    ifactor=10.0, dfactor=0.2, max_num_steps=2 ** 31 - 1, **unused_kwargs):
    _handle_unused_kwargs(self, unused_kwargs)
    del unused_kwargs
    self.func = func
    self.y0 = y0
    self.rtol = rtol
    self.atol = atol
    self.first_step = first_step
    self.safety = _convert_to_tensor(safety, dtype=torch.float64, device=y0
        [0].device)
    self.ifactor = _convert_to_tensor(ifactor, dtype=torch.float64, device=
        y0[0].device)
    self.dfactor = _convert_to_tensor(dfactor, dtype=torch.float64, device=
        y0[0].device)
    self.max_num_steps = _convert_to_tensor(max_num_steps, dtype=torch.
        int32, device=y0[0].device)
