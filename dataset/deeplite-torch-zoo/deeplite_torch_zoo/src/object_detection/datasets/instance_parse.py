def parse(x):
    """Parse bounding boxes format between XYWH and LTWH."""
    return x if isinstance(x, abc.Iterable) else tuple(repeat(x, n))
