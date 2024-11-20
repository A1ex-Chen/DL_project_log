def __getattr__(self, _):
    """Return empty function."""

    def empty_fn(*args, **kwargs):
        return
    return empty_fn
