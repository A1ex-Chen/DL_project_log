def _is_callable_dunder(maybe_fn):
    """
    Returns True if maybe_fn is a callable dunder (callable named with double
    underscores) (e.g., __add__)
    """
    return _is_callable(maybe_fn) and len(maybe_fn.__name__
        ) > 4 and maybe_fn.__name__[:2] == '__' and maybe_fn.__name__[-2:
        ] == '__' and maybe_fn.__name__ not in BLACKLISTED_DUNDERS
