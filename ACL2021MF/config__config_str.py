def _config_str(config: Config) ->str:
    """
    Collect a subset of config in sensible order (not alphabetical) according to phase. Used by
    :func:`Config.__str__()`.

    Parameters
    ----------
    config: Config
        A :class:`Config` object which is to be printed.
    """
    _C = config
    __C: CN = CN({'RANDOM_SEED': _C.random_seed})
    common_string: str = str(__C) + '\n'
    return common_string
