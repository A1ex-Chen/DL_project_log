@classmethod
def default_config_path(cls, type='default'):
    return utils.get_abs_path(cls.DATASET_CONFIG_DICT[type])
