def downgrade_config(cfg: CN, to_version: int) ->CN:
    """
    Downgrade a config from its current version to an older version.

    Args:
        cfg (CfgNode):
        to_version (int):

    Note:
        A general downgrade of arbitrary configs is not always possible due to the
        different functionalities in different versions.
        The purpose of downgrade is only to recover the defaults in old versions,
        allowing it to load an old partial yaml config.
        Therefore, the implementation only needs to fill in the default values
        in the old version when a general downgrade is not possible.
    """
    cfg = cfg.clone()
    assert cfg.VERSION >= to_version, 'Cannot downgrade from v{} to v{}!'.format(
        cfg.VERSION, to_version)
    for k in range(cfg.VERSION, to_version, -1):
        converter = globals()['ConverterV' + str(k)]
        converter.downgrade(cfg)
        cfg.VERSION = k - 1
    return cfg
