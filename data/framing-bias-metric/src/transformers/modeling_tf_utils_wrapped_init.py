@functools.wraps(initializer)
def wrapped_init(self, *args, **kwargs):
    config = args[0] if args and isinstance(args[0], PretrainedConfig
        ) else kwargs.pop('config', None)
    if isinstance(config, dict):
        config = config_class.from_dict(config)
        initializer(self, config, *args, **kwargs)
    elif isinstance(config, PretrainedConfig):
        if len(args) > 0:
            initializer(self, *args, **kwargs)
        else:
            initializer(self, config, *args, **kwargs)
    else:
        raise ValueError(
            'Must pass either `config` (PretrainedConfig) or `config` (dict)')
    self._config = config
    self._kwargs = kwargs
