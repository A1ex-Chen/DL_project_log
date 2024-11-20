@classmethod
def default_config_path(cls, model_type):
    assert model_type in cls.PRETRAINED_MODEL_CONFIG_DICT, 'Unknown model type {}'.format(
        model_type)
    return get_abs_path(cls.PRETRAINED_MODEL_CONFIG_DICT[model_type])
