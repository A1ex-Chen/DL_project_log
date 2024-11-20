def copy_attr(a, b, include=(), exclude=()):
    """Copy attributes from one instance and set them to another instance."""
    for k, item in b.__dict__.items():
        if len(include) and k not in include or k.startswith('_'
            ) or k in exclude:
            continue
        else:
            setattr(a, k, item)
