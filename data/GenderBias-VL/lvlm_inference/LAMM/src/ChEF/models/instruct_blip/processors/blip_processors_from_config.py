@classmethod
def from_config(cls, cfg=None):
    if cfg is None:
        cfg = OmegaConf.create()
    image_size = cfg.get('image_size', 364)
    mean = cfg.get('mean', None)
    std = cfg.get('std', None)
    min_scale = cfg.get('min_scale', 0.5)
    max_scale = cfg.get('max_scale', 1.0)
    return cls(image_size=image_size, mean=mean, std=std, min_scale=
        min_scale, max_scale=max_scale)
