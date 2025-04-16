def prepare_init_args_and_inputs_for_common(self):
    init_dict = {'block_out_channels': (4, 8), 'norm_num_groups': 4,
        'down_block_types': ('CrossAttnDownBlock3D', 'DownBlock3D'),
        'up_block_types': ('UpBlock3D', 'CrossAttnUpBlock3D'),
        'cross_attention_dim': 8, 'attention_head_dim': 2, 'out_channels': 
        4, 'in_channels': 4, 'layers_per_block': 1, 'sample_size': 16}
    inputs_dict = self.dummy_input
    return init_dict, inputs_dict
