def _m(*args, **kwargs):
    return Model(*args, constructor=EfficientNet, **kwargs)
