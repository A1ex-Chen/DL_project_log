def prepare_init_args_and_inputs_for_common(self):
    init_dict = {'block_out_channels': (32, 64), 'down_block_types': (
        'CrossAttnDownBlockSpatioTemporal', 'DownBlockSpatioTemporal'),
        'up_block_types': ('UpBlockSpatioTemporal',
        'CrossAttnUpBlockSpatioTemporal'), 'cross_attention_dim': 32,
        'num_attention_heads': 8, 'out_channels': 4, 'in_channels': 4,
        'layers_per_block': 2, 'sample_size': 32,
        'projection_class_embeddings_input_dim': self.
        addition_time_embed_dim * 3, 'addition_time_embed_dim': self.
        addition_time_embed_dim}
    inputs_dict = self.dummy_input
    return init_dict, inputs_dict
