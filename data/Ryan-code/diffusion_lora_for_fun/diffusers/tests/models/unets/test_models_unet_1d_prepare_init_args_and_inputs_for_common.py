def prepare_init_args_and_inputs_for_common(self):
    init_dict = {'in_channels': 14, 'out_channels': 14, 'down_block_types':
        ['DownResnetBlock1D', 'DownResnetBlock1D', 'DownResnetBlock1D',
        'DownResnetBlock1D'], 'up_block_types': [], 'out_block_type':
        'ValueFunction', 'mid_block_type': 'ValueFunctionMidBlock1D',
        'block_out_channels': [32, 64, 128, 256], 'layers_per_block': 1,
        'downsample_each_block': True, 'use_timestep_embedding': True,
        'freq_shift': 1.0, 'flip_sin_to_cos': False, 'time_embedding_type':
        'positional', 'act_fn': 'mish'}
    inputs_dict = self.dummy_input
    return init_dict, inputs_dict
