def get_dummy_components(self):
    torch.manual_seed(0)
    unet = UNet1DModel(block_out_channels=(32, 32, 64), extra_in_channels=
        16, sample_size=512, sample_rate=16000, in_channels=2, out_channels
        =2, flip_sin_to_cos=True, use_timestep_embedding=False,
        time_embedding_type='fourier', mid_block_type='UNetMidBlock1D',
        down_block_types=('DownBlock1DNoSkip', 'DownBlock1D',
        'AttnDownBlock1D'), up_block_types=('AttnUpBlock1D', 'UpBlock1D',
        'UpBlock1DNoSkip'))
    scheduler = IPNDMScheduler()
    components = {'unet': unet, 'scheduler': scheduler}
    return components
