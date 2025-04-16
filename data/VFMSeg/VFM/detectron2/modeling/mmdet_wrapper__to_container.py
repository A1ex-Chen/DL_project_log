def _to_container(cfg):
    """
    mmdet will assert the type of dict/list.
    So convert omegaconf objects to dict/list.
    """
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    from mmcv.utils import ConfigDict
    return ConfigDict(cfg)
