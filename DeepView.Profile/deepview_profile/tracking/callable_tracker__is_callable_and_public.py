def _is_callable_and_public(maybe_fn):
    return _is_callable(maybe_fn) and maybe_fn.__name__[0] != '_'
