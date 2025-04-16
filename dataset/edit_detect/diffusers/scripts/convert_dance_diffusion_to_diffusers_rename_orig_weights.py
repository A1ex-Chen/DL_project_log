def rename_orig_weights(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.endswith('kernel'):
            continue
        new_k = rename(k)
        if isinstance(new_k, list):
            new_state_dict = transform_conv_attns(new_state_dict, new_k, v)
        else:
            new_state_dict[new_k] = v
    return new_state_dict
