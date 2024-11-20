def get_up_block(up_block_type: str, num_layers: int, in_channels: int,
    out_channels: int, temb_channels: int, add_upsample: bool):
    deprecation_message = (
        'Importing `get_up_block` from `diffusers.models.unet_1d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_1d_blocks import get_up_block`, instead.'
        )
    deprecate('get_up_block', '0.29', deprecation_message)
    from .unets.unet_1d_blocks import get_up_block
    return get_up_block(up_block_type=up_block_type, num_layers=num_layers,
        in_channels=in_channels, out_channels=out_channels, temb_channels=
        temb_channels, add_upsample=add_upsample)
