def _find_mismatched_keys(state_dict, model_state_dict, loaded_keys,
    ignore_mismatched_sizes):
    mismatched_keys = []
    if ignore_mismatched_sizes:
        for checkpoint_key in loaded_keys:
            model_key = checkpoint_key
            if model_key in model_state_dict and state_dict[checkpoint_key
                ].shape != model_state_dict[model_key].shape:
                mismatched_keys.append((checkpoint_key, state_dict[
                    checkpoint_key].shape, model_state_dict[model_key].shape))
                del state_dict[checkpoint_key]
    return mismatched_keys
