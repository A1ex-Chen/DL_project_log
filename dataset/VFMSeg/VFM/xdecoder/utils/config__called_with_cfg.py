def _called_with_cfg(*args, **kwargs):
    """
    Returns:
        bool: whether the arguments contain CfgNode and should be considered
            forwarded to from_config.
    """
    from omegaconf import DictConfig
    if len(args) and isinstance(args[0], dict):
        return True
    if isinstance(kwargs.pop('cfg', None), dict):
        return True
    return False
