def requires_pytorch(obj):
    name = obj.__name__ if hasattr(obj, '__name__') else obj.__class__.__name__
    if not is_torch_available():
        raise ImportError(PYTORCH_IMPORT_ERROR.format(name))
