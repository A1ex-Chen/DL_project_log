def requires_flax(obj):
    name = obj.__name__ if hasattr(obj, '__name__') else obj.__class__.__name__
    if not is_flax_available():
        raise ImportError(FLAX_IMPORT_ERROR.format(name))
