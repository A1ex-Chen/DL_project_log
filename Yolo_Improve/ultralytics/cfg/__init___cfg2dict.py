def cfg2dict(cfg):
    """
    Convert a configuration object to a dictionary, whether it is a file path, a string, or a SimpleNamespace object.

    Args:
        cfg (str | Path | dict | SimpleNamespace): Configuration object to be converted to a dictionary. This may be a
            path to a configuration file, a dictionary, or a SimpleNamespace object.

    Returns:
        (dict): Configuration object in dictionary format.

    Example:
        ```python
        from ultralytics.cfg import cfg2dict
        from types import SimpleNamespace

        # Example usage with a file path
        config_dict = cfg2dict('config.yaml')

        # Example usage with a SimpleNamespace
        config_sn = SimpleNamespace(param1='value1', param2='value2')
        config_dict = cfg2dict(config_sn)

        # Example usage with a dictionary (returns the same dictionary)
        config_dict = cfg2dict({'param1': 'value1', 'param2': 'value2'})
        ```

    Notes:
        - If `cfg` is a path or a string, it will be loaded as YAML and converted to a dictionary.
        - If `cfg` is a SimpleNamespace object, it will be converted to a dictionary using `vars()`.
    """
    if isinstance(cfg, (str, Path)):
        cfg = yaml_load(cfg)
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)
    return cfg
