def extract_sub_model(checkpoint, key_prefix):
    """
    Takes a state dict and returns the state dict for a particular sub-model.
    """
    sub_model_state_dict = {}
    keys = list(checkpoint.keys())
    for key in keys:
        if key.startswith(key_prefix):
            sub_model_state_dict[key.replace(key_prefix, '')] = checkpoint.get(
                key)
    return sub_model_state_dict
