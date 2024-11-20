def get_mid_block(mid_block_type, num_layers, in_channels, mid_channels,
    out_channels, embed_dim, add_downsample):
    if mid_block_type == 'MidResTemporalBlock1D':
        return MidResTemporalBlock1D(num_layers=num_layers, in_channels=
            in_channels, out_channels=out_channels, embed_dim=embed_dim,
            add_downsample=add_downsample)
    elif mid_block_type == 'ValueFunctionMidBlock1D':
        return ValueFunctionMidBlock1D(in_channels=in_channels,
            out_channels=out_channels, embed_dim=embed_dim)
    elif mid_block_type == 'UNetMidBlock1D':
        return UNetMidBlock1D(in_channels=in_channels, mid_channels=
            mid_channels, out_channels=out_channels)
    raise ValueError(f'{mid_block_type} does not exist.')
