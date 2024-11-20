def requires_faiss(obj):
    name = obj.__name__ if hasattr(obj, '__name__') else obj.__class__.__name__
    if not is_faiss_available():
        raise ImportError(FAISS_IMPORT_ERROR.format(name))
