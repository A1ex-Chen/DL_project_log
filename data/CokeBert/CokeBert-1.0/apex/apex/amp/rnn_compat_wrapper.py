def wrapper(*args, **kwargs):
    return getattr(_VF, name)(*args, **kwargs)
