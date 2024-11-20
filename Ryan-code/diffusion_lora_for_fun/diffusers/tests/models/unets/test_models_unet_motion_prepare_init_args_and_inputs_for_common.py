def prepare_init_args_and_inputs_for_common(self):
    init_dict = {'block_out_channels': (16, 32), 'norm_num_groups': 16,
        'down_block_types': ('CrossAttnDownBlockMotion', 'DownBlockMotion'),
        'up_block_types': ('UpBlockMotion', 'CrossAttnUpBlockMotion'),
        'cross_attention_dim': 16, 'num_attention_heads': 2, 'out_channels':
        4, 'in_channels': 4, 'layers_per_block': 1, 'sample_size': 16}
    inputs_dict = self.dummy_input
    return init_dict, inputs_dict
