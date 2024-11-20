def _compare_model_configs(self, dict0, dict1):
    if dict0 == dict1:
        return True
    else:
        config0, meta_keys0 = self._remove_meta_keys(dict0)
        config1, meta_keys1 = self._remove_meta_keys(dict1)
        if config0 == config1:
            print(f'Warning !: Mismatch in keys {meta_keys0} and {meta_keys1}.'
                )
            return True
    return False
