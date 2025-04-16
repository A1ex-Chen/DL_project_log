def get_cfg(cfg: Union[str, Path, Dict, SimpleNamespace]=DEFAULT_CFG_DICT,
    overrides: Dict=None):
    """
    Load and merge configuration data from a file or dictionary, with optional overrides.

    Args:
        cfg (str | Path | dict | SimpleNamespace, optional): Configuration data source. Defaults to `DEFAULT_CFG_DICT`.
        overrides (dict | None, optional): Dictionary containing key-value pairs to override the base configuration.
            Defaults to None.

    Returns:
        (SimpleNamespace): Namespace containing the merged training arguments.

    Notes:
        - If both `cfg` and `overrides` are provided, the values in `overrides` will take precedence.
        - Special handling ensures alignment and correctness of the configuration, such as converting numeric `project`
          and `name` to strings and validating the configuration keys and values.

    Example:
        ```python
        from ultralytics.cfg import get_cfg

        # Load default configuration
        config = get_cfg()

        # Load from a custom file with overrides
        config = get_cfg('path/to/config.yaml', overrides={'epochs': 50, 'batch_size': 16})
        ```

        Configuration dictionary merged with overrides:
        ```python
        {'epochs': 50, 'batch_size': 16, ...}
        ```
    """
    cfg = cfg2dict(cfg)
    if overrides:
        overrides = cfg2dict(overrides)
        if 'save_dir' not in cfg:
            overrides.pop('save_dir', None)
        check_dict_alignment(cfg, overrides)
        cfg = {**cfg, **overrides}
    for k in ('project', 'name'):
        if k in cfg and isinstance(cfg[k], (int, float)):
            cfg[k] = str(cfg[k])
    if cfg.get('name') == 'model':
        cfg['name'] = cfg.get('model', '').split('.')[0]
        LOGGER.warning(
            f"WARNING ⚠️ 'name=model' automatically updated to 'name={cfg['name']}'."
            )
    check_cfg(cfg)
    return IterableSimpleNamespace(**cfg)
