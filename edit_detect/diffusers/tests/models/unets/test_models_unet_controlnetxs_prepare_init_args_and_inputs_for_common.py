def prepare_init_args_and_inputs_for_common(self):
    init_dict = {'sample_size': 16, 'down_block_types': ('DownBlock2D',
        'CrossAttnDownBlock2D'), 'up_block_types': ('CrossAttnUpBlock2D',
        'UpBlock2D'), 'block_out_channels': (4, 8), 'cross_attention_dim': 
        8, 'transformer_layers_per_block': 1, 'num_attention_heads': 2,
        'norm_num_groups': 4, 'upcast_attention': False,
        'ctrl_block_out_channels': [2, 4], 'ctrl_num_attention_heads': 4,
        'ctrl_max_norm_num_groups': 2,
        'ctrl_conditioning_embedding_out_channels': (2, 2)}
    inputs_dict = self.dummy_input
    return init_dict, inputs_dict
