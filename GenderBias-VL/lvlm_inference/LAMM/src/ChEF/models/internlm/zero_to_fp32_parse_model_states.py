def parse_model_states(files):
    zero_model_states = []
    for file in files:
        state_dict = torch.load(file, map_location=device)
        if BUFFER_NAMES not in state_dict:
            raise ValueError(f'{file} is not a model state checkpoint')
        buffer_names = state_dict[BUFFER_NAMES]
        if debug:
            print('Found buffers:', buffer_names)
        buffers = {k: v.float() for k, v in state_dict['module'].items() if
            k in buffer_names}
        param_shapes = state_dict[PARAM_SHAPES]
        param_names = []
        for s in param_shapes:
            for name in s.keys():
                param_names.append(name)
        frozen_param_shapes = state_dict.get(FROZEN_PARAM_SHAPES, None)
        if frozen_param_shapes is not None:
            if debug:
                print(f'Found frozen_param_shapes: {frozen_param_shapes}')
            param_names += list(frozen_param_shapes.keys())
        shared_params = [[k, v] for k, v in state_dict['shared_params'].items()
            ]
        ds_version = state_dict.get(DS_VERSION, None)
        frozen_param_fragments = state_dict.get(FROZEN_PARAM_FRAGMENTS, None)
        z_model_state = zero_model_state(buffers=buffers, param_shapes=
            param_shapes, shared_params=shared_params, ds_version=
            ds_version, frozen_param_shapes=frozen_param_shapes,
            frozen_param_fragments=frozen_param_fragments)
        zero_model_states.append(z_model_state)
    return zero_model_states
