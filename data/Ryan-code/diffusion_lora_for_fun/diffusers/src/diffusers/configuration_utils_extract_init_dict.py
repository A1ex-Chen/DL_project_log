@classmethod
def extract_init_dict(cls, config_dict, **kwargs):
    used_defaults = config_dict.get('_use_default_values', [])
    config_dict = {k: v for k, v in config_dict.items() if k not in
        used_defaults and k != '_use_default_values'}
    original_dict = dict(config_dict.items())
    expected_keys = cls._get_init_keys(cls)
    expected_keys.remove('self')
    if 'kwargs' in expected_keys:
        expected_keys.remove('kwargs')
    if hasattr(cls, '_flax_internal_args'):
        for arg in cls._flax_internal_args:
            expected_keys.remove(arg)
    if len(cls.ignore_for_config) > 0:
        expected_keys = expected_keys - set(cls.ignore_for_config)
    diffusers_library = importlib.import_module(__name__.split('.')[0])
    if cls.has_compatibles:
        compatible_classes = [c for c in cls._get_compatibles() if not
            isinstance(c, DummyObject)]
    else:
        compatible_classes = []
    expected_keys_comp_cls = set()
    for c in compatible_classes:
        expected_keys_c = cls._get_init_keys(c)
        expected_keys_comp_cls = expected_keys_comp_cls.union(expected_keys_c)
    expected_keys_comp_cls = expected_keys_comp_cls - cls._get_init_keys(cls)
    config_dict = {k: v for k, v in config_dict.items() if k not in
        expected_keys_comp_cls}
    orig_cls_name = config_dict.pop('_class_name', cls.__name__)
    if isinstance(orig_cls_name, str
        ) and orig_cls_name != cls.__name__ and hasattr(diffusers_library,
        orig_cls_name):
        orig_cls = getattr(diffusers_library, orig_cls_name)
        unexpected_keys_from_orig = cls._get_init_keys(orig_cls
            ) - expected_keys
        config_dict = {k: v for k, v in config_dict.items() if k not in
            unexpected_keys_from_orig}
    elif not isinstance(orig_cls_name, str) and not isinstance(orig_cls_name,
        (list, tuple)):
        raise ValueError(
            'Make sure that the `_class_name` is of type string or list of string (for custom pipelines).'
            )
    config_dict = {k: v for k, v in config_dict.items() if not k.startswith
        ('_')}
    init_dict = {}
    for key in expected_keys:
        if key in kwargs and key in config_dict:
            config_dict[key] = kwargs.pop(key)
        if key in kwargs:
            init_dict[key] = kwargs.pop(key)
        elif key in config_dict:
            init_dict[key] = config_dict.pop(key)
    if len(config_dict) > 0:
        logger.warning(
            f'The config attributes {config_dict} were passed to {cls.__name__}, but are not expected and will be ignored. Please verify your {cls.config_name} configuration file.'
            )
    passed_keys = set(init_dict.keys())
    if len(expected_keys - passed_keys) > 0:
        logger.info(
            f'{expected_keys - passed_keys} was not found in config. Values will be initialized to default values.'
            )
    unused_kwargs = {**config_dict, **kwargs}
    hidden_config_dict = {k: v for k, v in original_dict.items() if k not in
        init_dict}
    return init_dict, unused_kwargs, hidden_config_dict
