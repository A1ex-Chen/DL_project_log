def requires_tf(obj):
    name = obj.__name__ if hasattr(obj, '__name__') else obj.__class__.__name__
    if not is_tf_available():
        raise ImportError(TENSORFLOW_IMPORT_ERROR.format(name))
