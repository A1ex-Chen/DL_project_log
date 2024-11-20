def get_out_block(*, out_block_type: str, num_groups_out: int, embed_dim:
    int, out_channels: int, act_fn: str, fc_dim: int):
    deprecation_message = (
        'Importing `get_out_block` from `diffusers.models.unet_1d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_1d_blocks import get_out_block`, instead.'
        )
    deprecate('get_out_block', '0.29', deprecation_message)
    from .unets.unet_1d_blocks import get_out_block
    return get_out_block(out_block_type=out_block_type, num_groups_out=
        num_groups_out, embed_dim=embed_dim, out_channels=out_channels,
        act_fn=act_fn, fc_dim=fc_dim)
