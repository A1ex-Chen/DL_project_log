def get_config(self):
    cfg = super(cls, self).get_config()
    cfg['config'] = self._config.to_dict()
    cfg.update(self._kwargs)
    return cfg
