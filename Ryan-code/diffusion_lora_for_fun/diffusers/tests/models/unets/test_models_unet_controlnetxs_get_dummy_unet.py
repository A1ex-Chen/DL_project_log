def get_dummy_unet(self):
    """For some tests we also need the underlying UNet. For these, we'll build the UNetControlNetXSModel from the UNet and ControlNetXS-Adapter"""
    return UNet2DConditionModel(block_out_channels=(4, 8), layers_per_block
        =2, sample_size=16, in_channels=4, out_channels=4, down_block_types
        =('DownBlock2D', 'CrossAttnDownBlock2D'), up_block_types=(
        'CrossAttnUpBlock2D', 'UpBlock2D'), cross_attention_dim=8,
        norm_num_groups=4, use_linear_projection=True)
