def build(self, **kwargs):
    cfg = OmegaConf.create(kwargs)
    return self.from_config(cfg)
