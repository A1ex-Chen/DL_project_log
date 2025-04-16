@property
def dummy_uncond_unet(self):
    torch.manual_seed(0)
    model = UNet2DModel(block_out_channels=(4, 8), layers_per_block=1,
        norm_num_groups=4, sample_size=8, in_channels=3, out_channels=3,
        down_block_types=('DownBlock2D', 'AttnDownBlock2D'), up_block_types
        =('AttnUpBlock2D', 'UpBlock2D'))
    return model
