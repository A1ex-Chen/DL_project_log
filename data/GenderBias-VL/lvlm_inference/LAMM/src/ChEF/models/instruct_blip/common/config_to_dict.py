def to_dict(self):
    return OmegaConf.to_container(self.config)
