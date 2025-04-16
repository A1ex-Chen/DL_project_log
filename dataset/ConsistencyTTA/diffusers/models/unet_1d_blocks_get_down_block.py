def get_down_block(down_block_type, num_layers, in_channels, out_channels,
    temb_channels, add_downsample):
    if down_block_type == 'DownResnetBlock1D':
        return DownResnetBlock1D(in_channels=in_channels, num_layers=
            num_layers, out_channels=out_channels, temb_channels=
            temb_channels, add_downsample=add_downsample)
    elif down_block_type == 'DownBlock1D':
        return DownBlock1D(out_channels=out_channels, in_channels=in_channels)
    elif down_block_type == 'AttnDownBlock1D':
        return AttnDownBlock1D(out_channels=out_channels, in_channels=
            in_channels)
    elif down_block_type == 'DownBlock1DNoSkip':
        return DownBlock1DNoSkip(out_channels=out_channels, in_channels=
            in_channels)
    raise ValueError(f'{down_block_type} does not exist.')
