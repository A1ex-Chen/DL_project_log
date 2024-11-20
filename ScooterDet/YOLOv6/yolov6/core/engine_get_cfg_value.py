def get_cfg_value(cfg_dict, value_str, default_value):
    if value_str in cfg_dict:
        if isinstance(cfg_dict[value_str], list):
            return cfg_dict[value_str][0] if cfg_dict[value_str][0
                ] is not None else default_value
        else:
            return cfg_dict[value_str] if cfg_dict[value_str
                ] is not None else default_value
    else:
        return default_value
