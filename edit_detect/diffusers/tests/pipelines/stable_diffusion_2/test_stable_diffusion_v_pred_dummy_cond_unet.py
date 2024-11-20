@property
def dummy_cond_unet(self):
    torch.manual_seed(0)
    model = UNet2DConditionModel(block_out_channels=(32, 64),
        layers_per_block=2, sample_size=32, in_channels=4, out_channels=4,
        down_block_types=('DownBlock2D', 'CrossAttnDownBlock2D'),
        up_block_types=('CrossAttnUpBlock2D', 'UpBlock2D'),
        cross_attention_dim=32, attention_head_dim=(2, 4),
        use_linear_projection=True)
    return model
