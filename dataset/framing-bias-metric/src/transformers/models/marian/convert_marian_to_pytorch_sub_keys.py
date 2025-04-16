def sub_keys(self, layer_prefix):
    return [remove_prefix(k, layer_prefix) for k in self.state_dict if k.
        startswith(layer_prefix)]
