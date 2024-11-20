@property
def dummy_uncond_unet(self):
    torch.manual_seed(0)
    model = UNet2DModel(block_out_channels=(32, 64), layers_per_block=2,
        sample_size=32, in_channels=6, out_channels=3, down_block_types=(
        'DownBlock2D', 'AttnDownBlock2D'), up_block_types=('AttnUpBlock2D',
        'UpBlock2D'))
    return model
