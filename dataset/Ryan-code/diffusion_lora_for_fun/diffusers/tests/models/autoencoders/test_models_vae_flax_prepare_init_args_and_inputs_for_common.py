def prepare_init_args_and_inputs_for_common(self):
    init_dict = {'block_out_channels': [32, 64], 'in_channels': 3,
        'out_channels': 3, 'down_block_types': ['DownEncoderBlock2D',
        'DownEncoderBlock2D'], 'up_block_types': ['UpDecoderBlock2D',
        'UpDecoderBlock2D'], 'latent_channels': 4}
    inputs_dict = self.dummy_input
    return init_dict, inputs_dict
