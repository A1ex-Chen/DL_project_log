@classmethod
def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.
    PathLike], **kwargs) ->'PretrainedConfig':
    config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path,
        **kwargs)
    if config_dict.get('model_type') == 'mplug-owl':
        config_dict = config_dict['abstractor_config']
    if 'model_type' in config_dict and hasattr(cls, 'model_type'
        ) and config_dict['model_type'] != cls.model_type:
        logger.warning(
            f"You are using a model of type {config_dict['model_type']} to instantiate a model of type {cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )
    return cls.from_dict(config_dict, **kwargs)
