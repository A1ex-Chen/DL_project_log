def requires_protobuf(obj):
    name = obj.__name__ if hasattr(obj, '__name__') else obj.__class__.__name__
    if not is_protobuf_available():
        raise ImportError(PROTOBUF_IMPORT_ERROR.format(name))
