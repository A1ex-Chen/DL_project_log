def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.
    Args:
        cfg (edict): Config dict. It should at least contain the key "NAME".
        registry (:obj:`Registry`): The registry to search the type from.
    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'NAME' not in cfg:
        if default_args is None or 'NAME' not in default_args:
            raise KeyError(
                f"""`cfg` or `default_args` must contain the key "NAME", but got {cfg}
{default_args}"""
                )
    if not isinstance(registry, Registry):
        raise TypeError(
            f'registry must be an mmcv.Registry object, but got {type(registry)}'
            )
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError(
            f'default_args must be a dict or None, but got {type(default_args)}'
            )
    if default_args is not None:
        cfg = config.merge_new_config(cfg, default_args)
    obj_type = cfg.get('NAME')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f'{obj_type} is not in the {registry.name} registry'
                )
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            f'type must be a str or valid type, but got {type(obj_type)}')
    try:
        return obj_cls(cfg)
    except Exception as e:
        raise type(e)(f'{obj_cls.__name__}: {e}')
