def build(self, config_path):
    self.config = self.get_config(config_path)
    ret = self._build()
    self.built = True
    return ret
