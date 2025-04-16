@property
def dummy_movq_kwargs(self):
    return {'block_out_channels': [32, 32, 64, 64], 'down_block_types': [
        'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D',
        'AttnDownEncoderBlock2D'], 'in_channels': 3, 'latent_channels': 4,
        'layers_per_block': 1, 'norm_num_groups': 8, 'norm_type': 'spatial',
        'num_vq_embeddings': 12, 'out_channels': 3, 'up_block_types': [
        'AttnUpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D',
        'UpDecoderBlock2D'], 'vq_embed_dim': 4}
