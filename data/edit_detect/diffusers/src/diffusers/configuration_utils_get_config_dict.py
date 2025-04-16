@classmethod
def get_config_dict(cls, *args, **kwargs):
    deprecation_message = (
        f' The function get_config_dict is deprecated. Please use {cls}.load_config instead. This function will be removed in version v1.0.0'
        )
    deprecate('get_config_dict', '1.0.0', deprecation_message,
        standard_warn=False)
    return cls.load_config(*args, **kwargs)
