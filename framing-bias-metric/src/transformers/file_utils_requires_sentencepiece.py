def requires_sentencepiece(obj):
    name = obj.__name__ if hasattr(obj, '__name__') else obj.__class__.__name__
    if not is_sentencepiece_available():
        raise ImportError(SENTENCEPIECE_IMPORT_ERROR.format(name))
