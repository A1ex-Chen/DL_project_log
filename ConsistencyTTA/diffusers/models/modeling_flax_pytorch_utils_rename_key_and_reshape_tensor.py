def rename_key_and_reshape_tensor(pt_tuple_key, pt_tensor,
    random_flax_state_dict):
    """Rename PT weight names to corresponding Flax weight names and reshape tensor if necessary"""
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ('scale',)
    if any('norm' in str_ for str_ in pt_tuple_key) and pt_tuple_key[-1
        ] == 'bias' and pt_tuple_key[:-1] + ('bias',
        ) not in random_flax_state_dict and pt_tuple_key[:-1] + ('scale',
        ) in random_flax_state_dict:
        renamed_pt_tuple_key = pt_tuple_key[:-1] + ('scale',)
        return renamed_pt_tuple_key, pt_tensor
    elif pt_tuple_key[-1] in ['weight', 'gamma'] and pt_tuple_key[:-1] + (
        'scale',) in random_flax_state_dict:
        renamed_pt_tuple_key = pt_tuple_key[:-1] + ('scale',)
        return renamed_pt_tuple_key, pt_tensor
    if pt_tuple_key[-1] == 'weight' and pt_tuple_key[:-1] + ('embedding',
        ) in random_flax_state_dict:
        pt_tuple_key = pt_tuple_key[:-1] + ('embedding',)
        return renamed_pt_tuple_key, pt_tensor
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ('kernel',)
    if pt_tuple_key[-1] == 'weight' and pt_tensor.ndim == 4:
        pt_tensor = pt_tensor.transpose(2, 3, 1, 0)
        return renamed_pt_tuple_key, pt_tensor
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ('kernel',)
    if pt_tuple_key[-1] == 'weight':
        pt_tensor = pt_tensor.T
        return renamed_pt_tuple_key, pt_tensor
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ('weight',)
    if pt_tuple_key[-1] == 'gamma':
        return renamed_pt_tuple_key, pt_tensor
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ('bias',)
    if pt_tuple_key[-1] == 'beta':
        return renamed_pt_tuple_key, pt_tensor
    return pt_tuple_key, pt_tensor