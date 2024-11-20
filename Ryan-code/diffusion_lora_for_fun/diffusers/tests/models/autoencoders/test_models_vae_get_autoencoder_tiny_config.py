def get_autoencoder_tiny_config(block_out_channels=None):
    block_out_channels = len(block_out_channels) * [32
        ] if block_out_channels is not None else [32, 32]
    init_dict = {'in_channels': 3, 'out_channels': 3,
        'encoder_block_out_channels': block_out_channels,
        'decoder_block_out_channels': block_out_channels,
        'num_encoder_blocks': [(b // min(block_out_channels)) for b in
        block_out_channels], 'num_decoder_blocks': [(b // min(
        block_out_channels)) for b in reversed(block_out_channels)]}
    return init_dict
