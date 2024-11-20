def instantiate(cfg):
    """
    Recursively instantiate objects defined in dictionaries by
    "_target_" and arguments.

    Args:
        cfg: a dict-like object with "_target_" that defines the caller, and
            other keys that define the arguments

    Returns:
        object instantiated by cfg
    """
    from omegaconf import ListConfig
    if isinstance(cfg, ListConfig):
        lst = [instantiate(x) for x in cfg]
        return ListConfig(lst, flags={'allow_objects': True})
    if isinstance(cfg, list):
        return [instantiate(x) for x in cfg]
    if isinstance(cfg, abc.Mapping) and '_target_' in cfg:
        cfg = {k: instantiate(v) for k, v in cfg.items()}
        cls = cfg.pop('_target_')
        cls = instantiate(cls)
        if isinstance(cls, str):
            cls_name = cls
            cls = locate(cls_name)
            assert cls is not None, cls_name
        else:
            try:
                cls_name = cls.__module__ + '.' + cls.__qualname__
            except Exception:
                cls_name = str(cls)
        assert callable(cls
            ), f'_target_ {cls} does not define a callable object'
        try:
            return cls(**cfg)
        except TypeError:
            logger = logging.getLogger(__name__)
            logger.error(f'Error when instantiating {cls_name}!')
            raise
    return cfg
