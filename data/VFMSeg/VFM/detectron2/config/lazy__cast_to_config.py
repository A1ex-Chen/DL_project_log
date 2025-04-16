def _cast_to_config(obj):
    if isinstance(obj, dict):
        return DictConfig(obj, flags={'allow_objects': True})
    return obj
