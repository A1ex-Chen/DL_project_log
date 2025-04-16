def copy_attr(a, b, include=(), exclude=()):
    for k, v in b.__dict__.items():
        if len(include) and k not in include or k.startswith('_'
            ) or k in exclude:
            continue
        else:
            setattr(a, k, v)
