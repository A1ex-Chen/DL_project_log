def keras_serializable(cls):
    """
    Decorate a Keras Layer class to support Keras serialization.

    This is done by:

    1. Adding a :obj:`transformers_config` dict to the Keras config dictionary in :obj:`get_config` (called by Keras at
       serialization time.
    2. Wrapping :obj:`__init__` to accept that :obj:`transformers_config` dict (passed by Keras at deserialization
       time) and convert it to a config object for the actual layer initializer.
    3. Registering the class as a custom object in Keras (if the Tensorflow version supports this), so that it does not
       need to be supplied in :obj:`custom_objects` in the call to :obj:`tf.keras.models.load_model`.

    Args:
        cls (a :obj:`tf.keras.layers.Layers subclass`):
            Typically a :obj:`TF.MainLayer` class in this project, in general must accept a :obj:`config` argument to
            its initializer.

    Returns:
        The same class object, with modifications for Keras deserialization.
    """
    initializer = cls.__init__
    config_class = getattr(cls, 'config_class', None)
    if config_class is None:
        raise AttributeError(
            'Must set `config_class` to use @keras_serializable')

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
                'Must pass either `config` (PretrainedConfig) or `config` (dict)'
                )
        self._config = config
        self._kwargs = kwargs
    cls.__init__ = wrapped_init
    if not hasattr(cls, 'get_config'):
        raise TypeError(
            'Only use @keras_serializable on tf.keras.layers.Layer subclasses')
    if hasattr(cls.get_config, '_is_default'):

        def get_config(self):
            cfg = super(cls, self).get_config()
            cfg['config'] = self._config.to_dict()
            cfg.update(self._kwargs)
            return cfg
        cls.get_config = get_config
    cls._keras_serializable = True
    if hasattr(tf.keras.utils, 'register_keras_serializable'):
        cls = tf.keras.utils.register_keras_serializable()(cls)
    return cls
