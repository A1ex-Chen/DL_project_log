def _get_dim(dim):
    which = dim.WhichOneof('value')
    if which is not None:
        dim = getattr(dim, which)
    return None if isinstance(dim, (str, bytes)) else dim
