def upgrade_config(cfg: CN, to_version: Optional[int]=None) ->CN:
    """
    Upgrade a config from its current version to a newer version.

    Args:
        cfg (CfgNode):
        to_version (int): defaults to the latest version.
    """
    cfg = cfg.clone()
    if to_version is None:
        to_version = _C.VERSION
    assert cfg.VERSION <= to_version, 'Cannot upgrade from v{} to v{}!'.format(
        cfg.VERSION, to_version)
    for k in range(cfg.VERSION, to_version):
        converter = globals()['ConverterV' + str(k + 1)]
        converter.upgrade(cfg)
        cfg.VERSION = k + 1
    return cfg
