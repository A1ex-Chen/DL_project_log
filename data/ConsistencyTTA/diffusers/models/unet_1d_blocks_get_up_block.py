def get_up_block(up_block_type, num_layers, in_channels, out_channels,
    temb_channels, add_upsample):
    if up_block_type == 'UpResnetBlock1D':
        return UpResnetBlock1D(in_channels=in_channels, num_layers=
            num_layers, out_channels=out_channels, temb_channels=
            temb_channels, add_upsample=add_upsample)
    elif up_block_type == 'UpBlock1D':
        return UpBlock1D(in_channels=in_channels, out_channels=out_channels)
    elif up_block_type == 'AttnUpBlock1D':
        return AttnUpBlock1D(in_channels=in_channels, out_channels=out_channels
            )
    elif up_block_type == 'UpBlock1DNoSkip':
        return UpBlock1DNoSkip(in_channels=in_channels, out_channels=
            out_channels)
    raise ValueError(f'{up_block_type} does not exist.')
