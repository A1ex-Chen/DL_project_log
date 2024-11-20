def _zero3_merge_frozen_params(state_dict, world_size, zero_model_states):
    if zero_model_states[0].frozen_param_shapes is None or len(
        zero_model_states[0].frozen_param_shapes) == 0:
        return
    if debug:
        for i in range(world_size):
            num_elem = sum(s.numel() for s in zero_model_states[i].
                frozen_param_fragments.values())
            print(f'rank {i}: {FROZEN_PARAM_SHAPES}.numel = {num_elem}')
        frozen_param_shapes = zero_model_states[0].frozen_param_shapes
        wanted_params = len(frozen_param_shapes)
        wanted_numel = sum(s.numel() for s in frozen_param_shapes.values())
        avail_numel = sum([p.numel() for p in zero_model_states[0].
            frozen_param_fragments.values()]) * world_size
        print(f'Frozen params: Have {avail_numel} numels to process.')
        print(
            f'Frozen params: Need {wanted_numel} numels in {wanted_params} params'
            )
    total_params = 0
    total_numel = 0
    for name, shape in zero_model_states[0].frozen_param_shapes.items():
        total_params += 1
        unpartitioned_numel = shape.numel()
        total_numel += unpartitioned_numel
        param_frags = tuple(model_state.frozen_param_fragments[name] for
            model_state in zero_model_states)
        state_dict[name] = torch.cat(param_frags, 0).narrow(0, 0,
            unpartitioned_numel).view(shape)
        partitioned_numel, partitioned_padding_numel = (
            zero3_partitioned_param_info(unpartitioned_numel, world_size))
        if debug:
            print(
                f'Frozen params: {total_params} {name} full shape: {shape} partition0 numel={partitioned_numel} partitioned_padding_numel={partitioned_padding_numel}'
                )
    print(
        f'Reconstructed Frozen fp32 state dict with {total_params} params {total_numel} elements'
        )
