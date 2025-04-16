def remap_key(key, state_dict):
    for k in self.split_keys:
        if k in key:
            return key.split(k)[0] + k
    raise ValueError(
        f'There seems to be a problem with the state_dict: {set(state_dict.keys())}. {key} has to have one of {self.split_keys}.'
        )
