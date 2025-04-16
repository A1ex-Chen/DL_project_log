def _visit_dict_config(cfg, func):
    """
    Apply func recursively to all DictConfig in cfg.
    """
    if isinstance(cfg, DictConfig):
        func(cfg)
        for v in cfg.values():
            _visit_dict_config(v, func)
    elif isinstance(cfg, ListConfig):
        for v in cfg:
            _visit_dict_config(v, func)
