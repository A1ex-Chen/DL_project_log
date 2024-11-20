def _is_callable(maybe_fn):
    return inspect.isfunction(maybe_fn) or inspect.ismethod(maybe_fn
        ) or inspect.isbuiltin(maybe_fn) or inspect.isroutine(maybe_fn)
