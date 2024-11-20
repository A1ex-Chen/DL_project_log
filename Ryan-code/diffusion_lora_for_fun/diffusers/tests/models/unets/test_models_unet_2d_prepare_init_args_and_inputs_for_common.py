def prepare_init_args_and_inputs_for_common(self):
    init_dict = {'block_out_channels': [32, 64, 64, 64], 'in_channels': 3,
        'layers_per_block': 1, 'out_channels': 3, 'time_embedding_type':
        'fourier', 'norm_eps': 1e-06, 'mid_block_scale_factor': math.sqrt(
        2.0), 'norm_num_groups': None, 'down_block_types': [
        'SkipDownBlock2D', 'AttnSkipDownBlock2D', 'SkipDownBlock2D',
        'SkipDownBlock2D'], 'up_block_types': ['SkipUpBlock2D',
        'SkipUpBlock2D', 'AttnSkipUpBlock2D', 'SkipUpBlock2D']}
    inputs_dict = self.dummy_input
    return init_dict, inputs_dict
