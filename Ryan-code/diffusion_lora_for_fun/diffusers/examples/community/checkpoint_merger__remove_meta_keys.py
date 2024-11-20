def _remove_meta_keys(self, config_dict: Dict):
    meta_keys = []
    temp_dict = config_dict.copy()
    for key in config_dict.keys():
        if key.startswith('_'):
            temp_dict.pop(key)
            meta_keys.append(key)
    return temp_dict, meta_keys
