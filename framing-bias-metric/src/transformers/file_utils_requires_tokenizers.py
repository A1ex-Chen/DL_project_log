def requires_tokenizers(obj):
    name = obj.__name__ if hasattr(obj, '__name__') else obj.__class__.__name__
    if not is_tokenizers_available():
        raise ImportError(TOKENIZERS_IMPORT_ERROR.format(name))
