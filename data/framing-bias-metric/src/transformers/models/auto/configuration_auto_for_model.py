@classmethod
def for_model(cls, model_type: str, *args, **kwargs):
    if model_type in CONFIG_MAPPING:
        config_class = CONFIG_MAPPING[model_type]
        return config_class(*args, **kwargs)
    raise ValueError(
        'Unrecognized model identifier: {}. Should contain one of {}'.
        format(model_type, ', '.join(CONFIG_MAPPING.keys())))
