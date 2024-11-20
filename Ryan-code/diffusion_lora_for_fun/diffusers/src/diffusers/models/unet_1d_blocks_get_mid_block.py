def get_mid_block(mid_block_type: str, num_layers: int, in_channels: int,
    mid_channels: int, out_channels: int, embed_dim: int, add_downsample: bool
    ):
    deprecation_message = (
        'Importing `get_mid_block` from `diffusers.models.unet_1d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_1d_blocks import get_mid_block`, instead.'
        )
    deprecate('get_mid_block', '0.29', deprecation_message)
    from .unets.unet_1d_blocks import get_mid_block
    return get_mid_block(mid_block_type=mid_block_type, num_layers=
        num_layers, in_channels=in_channels, mid_channels=mid_channels,
        out_channels=out_channels, embed_dim=embed_dim, add_downsample=
        add_downsample)
