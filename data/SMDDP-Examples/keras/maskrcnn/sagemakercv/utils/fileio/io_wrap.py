def wrap(cls):
    _register_handler(cls(**kwargs), file_formats)
    return cls
