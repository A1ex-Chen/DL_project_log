def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
    missing_keys, unexpected_keys, error_msgs):
    version = local_metadata.get('version', None)
    if version is None or version < 2:
        if prefix + 'running_mean' not in state_dict:
            state_dict[prefix + 'running_mean'] = torch.zeros_like(self.
                running_mean)
        if prefix + 'running_var' not in state_dict:
            state_dict[prefix + 'running_var'] = torch.ones_like(self.
                running_var)
    super()._load_from_state_dict(state_dict, prefix, local_metadata,
        strict, missing_keys, unexpected_keys, error_msgs)
