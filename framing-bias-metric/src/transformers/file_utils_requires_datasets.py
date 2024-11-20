def requires_datasets(obj):
    name = obj.__name__ if hasattr(obj, '__name__') else obj.__class__.__name__
    if not is_datasets_available():
        raise ImportError(DATASETS_IMPORT_ERROR.format(name))
