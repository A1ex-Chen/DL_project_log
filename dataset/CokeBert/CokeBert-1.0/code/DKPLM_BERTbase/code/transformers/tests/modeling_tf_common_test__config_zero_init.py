def _config_zero_init(config):
    configs_no_init = copy.deepcopy(config)
    for key in configs_no_init.__dict__.keys():
        if '_range' in key or '_std' in key:
            setattr(configs_no_init, key, 0.0)
    return configs_no_init
