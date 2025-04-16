def _zero2_merge_frozen_params(state_dict, zero_model_states):
    if zero_model_states[0].frozen_param_shapes is None or len(
        zero_model_states[0].frozen_param_shapes) == 0:
        return
    frozen_param_shapes = zero_model_states[0].frozen_param_shapes
    frozen_param_fragments = zero_model_states[0].frozen_param_fragments
    if debug:
        num_elem = sum(s.numel() for s in frozen_param_shapes.values())
        print(f'rank 0: {FROZEN_PARAM_SHAPES}.numel = {num_elem}')
        wanted_params = len(frozen_param_shapes)
        wanted_numel = sum(s.numel() for s in frozen_param_shapes.values())
        avail_numel = sum([p.numel() for p in frozen_param_fragments.values()])
        print(f'Frozen params: Have {avail_numel} numels to process.')
        print(
            f'Frozen params: Need {wanted_numel} numels in {wanted_params} params'
            )
    total_params = 0
    total_numel = 0
    for name, shape in frozen_param_shapes.items():
        total_params += 1
        unpartitioned_numel = shape.numel()
        total_numel += unpartitioned_numel
        state_dict[name] = frozen_param_fragments[name]
        if debug:
            print(
                f'{name} full shape: {shape} unpartitioned numel {unpartitioned_numel} '
                )
    print(
        f'Reconstructed Frozen fp32 state dict with {total_params} params {total_numel} elements'
        )
