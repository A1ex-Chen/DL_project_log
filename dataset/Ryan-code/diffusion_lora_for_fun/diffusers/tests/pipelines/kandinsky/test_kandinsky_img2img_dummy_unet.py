@property
def dummy_unet(self):
    torch.manual_seed(0)
    model_kwargs = {'in_channels': 4, 'out_channels': 8,
        'addition_embed_type': 'text_image', 'down_block_types': (
        'ResnetDownsampleBlock2D', 'SimpleCrossAttnDownBlock2D'),
        'up_block_types': ('SimpleCrossAttnUpBlock2D',
        'ResnetUpsampleBlock2D'), 'mid_block_type':
        'UNetMidBlock2DSimpleCrossAttn', 'block_out_channels': (self.
        block_out_channels_0, self.block_out_channels_0 * 2),
        'layers_per_block': 1, 'encoder_hid_dim': self.
        text_embedder_hidden_size, 'encoder_hid_dim_type':
        'text_image_proj', 'cross_attention_dim': self.cross_attention_dim,
        'attention_head_dim': 4, 'resnet_time_scale_shift': 'scale_shift',
        'class_embed_type': None}
    model = UNet2DConditionModel(**model_kwargs)
    return model
