def get_config_dict(self):
    return {key: self.__dict__[key] for key in self.config_keys}
