@property
def dummy_decoder(self):
    torch.manual_seed(0)
    model_kwargs = {'sample_size': 32, 'in_channels': 3, 'out_channels': 6,
        'down_block_types': ('ResnetDownsampleBlock2D',
        'SimpleCrossAttnDownBlock2D'), 'up_block_types': (
        'SimpleCrossAttnUpBlock2D', 'ResnetUpsampleBlock2D'),
        'mid_block_type': 'UNetMidBlock2DSimpleCrossAttn',
        'block_out_channels': (self.block_out_channels_0, self.
        block_out_channels_0 * 2), 'layers_per_block': 1,
        'cross_attention_dim': self.cross_attention_dim,
        'attention_head_dim': 4, 'resnet_time_scale_shift': 'scale_shift',
        'class_embed_type': 'identity'}
    model = UNet2DConditionModel(**model_kwargs)
    return model
