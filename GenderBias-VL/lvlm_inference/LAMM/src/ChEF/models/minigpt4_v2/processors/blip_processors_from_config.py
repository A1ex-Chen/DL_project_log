@classmethod
def from_config(cls, cfg=None):
    if cfg is None:
        cfg = OmegaConf.create()
    image_size = cfg.get('image_size', 224)
    mean = cfg.get('mean', None)
    std = cfg.get('std', None)
    return cls(image_size=image_size, mean=mean, std=std)
