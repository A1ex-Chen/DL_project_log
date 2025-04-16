def wrapper(*args, **kwargs):
    kwargs['device'] = device
    return fn(*args, **kwargs)
