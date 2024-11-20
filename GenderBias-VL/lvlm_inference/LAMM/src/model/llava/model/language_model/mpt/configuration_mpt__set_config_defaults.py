def _set_config_defaults(self, config, config_defaults):
    for k, v in config_defaults.items():
        if k not in config:
            config[k] = v
    return config
