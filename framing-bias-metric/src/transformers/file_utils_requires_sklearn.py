def requires_sklearn(obj):
    name = obj.__name__ if hasattr(obj, '__name__') else obj.__class__.__name__
    if not is_sklearn_available():
        raise ImportError(SKLEARN_IMPORT_ERROR.format(name))
