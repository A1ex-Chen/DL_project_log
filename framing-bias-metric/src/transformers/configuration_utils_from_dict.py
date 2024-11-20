@classmethod
def from_dict(cls, config_dict: Dict[str, Any], **kwargs) ->'PretrainedConfig':
    """
        Instantiates a :class:`~transformers.PretrainedConfig` from a Python dictionary of parameters.

        Args:
            config_dict (:obj:`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                :func:`~transformers.PretrainedConfig.get_config_dict` method.
            kwargs (:obj:`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            :class:`PretrainedConfig`: The configuration object instantiated from those parameters.
        """
    return_unused_kwargs = kwargs.pop('return_unused_kwargs', False)
    config = cls(**config_dict)
    if hasattr(config, 'pruned_heads'):
        config.pruned_heads = dict((int(key), value) for key, value in
            config.pruned_heads.items())
    to_remove = []
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
            to_remove.append(key)
    for key in to_remove:
        kwargs.pop(key, None)
    logger.info('Model config %s', str(config))
    if return_unused_kwargs:
        return config, kwargs
    else:
        return config
