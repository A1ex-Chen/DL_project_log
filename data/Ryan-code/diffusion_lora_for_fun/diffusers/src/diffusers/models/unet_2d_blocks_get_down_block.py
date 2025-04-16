def get_down_block(down_block_type: str, num_layers: int, in_channels: int,
    out_channels: int, temb_channels: int, add_downsample: bool, resnet_eps:
    float, resnet_act_fn: str, transformer_layers_per_block: int=1,
    num_attention_heads: Optional[int]=None, resnet_groups: Optional[int]=
    None, cross_attention_dim: Optional[int]=None, downsample_padding:
    Optional[int]=None, dual_cross_attention: bool=False,
    use_linear_projection: bool=False, only_cross_attention: bool=False,
    upcast_attention: bool=False, resnet_time_scale_shift: str='default',
    attention_type: str='default', resnet_skip_time_act: bool=False,
    resnet_out_scale_factor: float=1.0, cross_attention_norm: Optional[str]
    =None, attention_head_dim: Optional[int]=None, downsample_type:
    Optional[str]=None, dropout: float=0.0):
    deprecation_message = (
        'Importing `get_down_block` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import get_down_block`, instead.'
        )
    deprecate('get_down_block', '0.29', deprecation_message)
    from .unets.unet_2d_blocks import get_down_block
    return get_down_block(down_block_type=down_block_type, num_layers=
        num_layers, in_channels=in_channels, out_channels=out_channels,
        temb_channels=temb_channels, add_downsample=add_downsample,
        resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn,
        transformer_layers_per_block=transformer_layers_per_block,
        num_attention_heads=num_attention_heads, resnet_groups=
        resnet_groups, cross_attention_dim=cross_attention_dim,
        downsample_padding=downsample_padding, dual_cross_attention=
        dual_cross_attention, use_linear_projection=use_linear_projection,
        only_cross_attention=only_cross_attention, upcast_attention=
        upcast_attention, resnet_time_scale_shift=resnet_time_scale_shift,
        attention_type=attention_type, resnet_skip_time_act=
        resnet_skip_time_act, resnet_out_scale_factor=
        resnet_out_scale_factor, cross_attention_norm=cross_attention_norm,
        attention_head_dim=attention_head_dim, downsample_type=
        downsample_type, dropout=dropout)
