def get_mid_block(mid_block_type: str, temb_channels: int, in_channels: int,
    resnet_eps: float, resnet_act_fn: str, resnet_groups: int,
    output_scale_factor: float=1.0, transformer_layers_per_block: int=1,
    num_attention_heads: Optional[int]=None, cross_attention_dim: Optional[
    int]=None, dual_cross_attention: bool=False, use_linear_projection:
    bool=False, mid_block_only_cross_attention: bool=False,
    upcast_attention: bool=False, resnet_time_scale_shift: str='default',
    attention_type: str='default', resnet_skip_time_act: bool=False,
    cross_attention_norm: Optional[str]=None, attention_head_dim: Optional[
    int]=1, dropout: float=0.0):
    if mid_block_type == 'UNetMidBlock2DCrossAttn':
        return UNetMidBlock2DCrossAttn(transformer_layers_per_block=
            transformer_layers_per_block, in_channels=in_channels,
            temb_channels=temb_channels, dropout=dropout, resnet_eps=
            resnet_eps, resnet_act_fn=resnet_act_fn, output_scale_factor=
            output_scale_factor, resnet_time_scale_shift=
            resnet_time_scale_shift, cross_attention_dim=
            cross_attention_dim, num_attention_heads=num_attention_heads,
            resnet_groups=resnet_groups, dual_cross_attention=
            dual_cross_attention, use_linear_projection=
            use_linear_projection, upcast_attention=upcast_attention,
            attention_type=attention_type)
    elif mid_block_type == 'UNetMidBlock2DSimpleCrossAttn':
        return UNetMidBlock2DSimpleCrossAttn(in_channels=in_channels,
            temb_channels=temb_channels, dropout=dropout, resnet_eps=
            resnet_eps, resnet_act_fn=resnet_act_fn, output_scale_factor=
            output_scale_factor, cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim, resnet_groups=
            resnet_groups, resnet_time_scale_shift=resnet_time_scale_shift,
            skip_time_act=resnet_skip_time_act, only_cross_attention=
            mid_block_only_cross_attention, cross_attention_norm=
            cross_attention_norm)
    elif mid_block_type == 'UNetMidBlock2D':
        return UNetMidBlock2D(in_channels=in_channels, temb_channels=
            temb_channels, dropout=dropout, num_layers=0, resnet_eps=
            resnet_eps, resnet_act_fn=resnet_act_fn, output_scale_factor=
            output_scale_factor, resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift, add_attention=
            False)
    elif mid_block_type is None:
        return None
    else:
        raise ValueError(f'unknown mid_block_type : {mid_block_type}')
