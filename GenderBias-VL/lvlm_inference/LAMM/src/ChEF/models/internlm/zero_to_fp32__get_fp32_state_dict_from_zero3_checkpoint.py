def _get_fp32_state_dict_from_zero3_checkpoint(world_size, fp32_flat_groups,
    zero_model_states):
    state_dict = OrderedDict()
    buffers = zero_model_states[0].buffers
    state_dict.update(buffers)
    if debug:
        print(f'added {len(buffers)} buffers')
    _zero3_merge_frozen_params(state_dict, world_size, zero_model_states)
    _zero3_merge_trainable_params(state_dict, world_size, fp32_flat_groups,
        zero_model_states)
    for pair in zero_model_states[0].shared_params:
        if pair[1] in state_dict:
            state_dict[pair[0]] = state_dict[pair[1]]
    return state_dict
