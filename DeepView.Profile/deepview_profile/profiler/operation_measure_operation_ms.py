def measure_operation_ms(self, func, args, kwargs):
    for_inplace = _is_potentially_inplace(func.__name__)
    forward_args, forward_kwargs = self._get_args_for_profiling(args,
        kwargs, for_inplace)

    def forward_runnable():
        func(*forward_args, **forward_kwargs)
    forward_ms = self._measure_ms(forward_runnable)
    backward_args, backward_kwargs = self._get_args_for_profiling(args,
        kwargs, for_inplace)
    retval = func(*backward_args, **backward_kwargs)
    if not backward_available(retval):
        return forward_ms, None
    return forward_ms, self._measure_backward_ms(func.__name__, retval)
