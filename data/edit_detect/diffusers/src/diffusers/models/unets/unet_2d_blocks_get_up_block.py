def get_up_block(up_block_type: str, num_layers: int, in_channels: int,
    out_channels: int, prev_output_channel: int, temb_channels: int,
    add_upsample: bool, resnet_eps: float, resnet_act_fn: str,
    resolution_idx: Optional[int]=None, transformer_layers_per_block: int=1,
    num_attention_heads: Optional[int]=None, resnet_groups: Optional[int]=
    None, cross_attention_dim: Optional[int]=None, dual_cross_attention:
    bool=False, use_linear_projection: bool=False, only_cross_attention:
    bool=False, upcast_attention: bool=False, resnet_time_scale_shift: str=
    'default', attention_type: str='default', resnet_skip_time_act: bool=
    False, resnet_out_scale_factor: float=1.0, cross_attention_norm:
    Optional[str]=None, attention_head_dim: Optional[int]=None,
    upsample_type: Optional[str]=None, dropout: float=0.0) ->nn.Module:
    if attention_head_dim is None:
        logger.warning(
            f'It is recommended to provide `attention_head_dim` when calling `get_up_block`. Defaulting `attention_head_dim` to {num_attention_heads}.'
            )
        attention_head_dim = num_attention_heads
    up_block_type = up_block_type[7:] if up_block_type.startswith('UNetRes'
        ) else up_block_type
    if up_block_type == 'UpBlock2D':
        return UpBlock2D(num_layers=num_layers, in_channels=in_channels,
            out_channels=out_channels, prev_output_channel=
            prev_output_channel, temb_channels=temb_channels,
            resolution_idx=resolution_idx, dropout=dropout, add_upsample=
            add_upsample, resnet_eps=resnet_eps, resnet_act_fn=
            resnet_act_fn, resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift)
    elif up_block_type == 'ResnetUpsampleBlock2D':
        return ResnetUpsampleBlock2D(num_layers=num_layers, in_channels=
            in_channels, out_channels=out_channels, prev_output_channel=
            prev_output_channel, temb_channels=temb_channels,
            resolution_idx=resolution_idx, dropout=dropout, add_upsample=
            add_upsample, resnet_eps=resnet_eps, resnet_act_fn=
            resnet_act_fn, resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift, skip_time_act=
            resnet_skip_time_act, output_scale_factor=resnet_out_scale_factor)
    elif up_block_type == 'CrossAttnUpBlock2D':
        if cross_attention_dim is None:
            raise ValueError(
                'cross_attention_dim must be specified for CrossAttnUpBlock2D')
        return CrossAttnUpBlock2D(num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels, out_channels=out_channels,
            prev_output_channel=prev_output_channel, temb_channels=
            temb_channels, resolution_idx=resolution_idx, dropout=dropout,
            add_upsample=add_upsample, resnet_eps=resnet_eps, resnet_act_fn
            =resnet_act_fn, resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim, num_attention_heads=
            num_attention_heads, dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention, upcast_attention=
            upcast_attention, resnet_time_scale_shift=
            resnet_time_scale_shift, attention_type=attention_type)
    elif up_block_type == 'SimpleCrossAttnUpBlock2D':
        if cross_attention_dim is None:
            raise ValueError(
                'cross_attention_dim must be specified for SimpleCrossAttnUpBlock2D'
                )
        return SimpleCrossAttnUpBlock2D(num_layers=num_layers, in_channels=
            in_channels, out_channels=out_channels, prev_output_channel=
            prev_output_channel, temb_channels=temb_channels,
            resolution_idx=resolution_idx, dropout=dropout, add_upsample=
            add_upsample, resnet_eps=resnet_eps, resnet_act_fn=
            resnet_act_fn, resnet_groups=resnet_groups, cross_attention_dim
            =cross_attention_dim, attention_head_dim=attention_head_dim,
            resnet_time_scale_shift=resnet_time_scale_shift, skip_time_act=
            resnet_skip_time_act, output_scale_factor=
            resnet_out_scale_factor, only_cross_attention=
            only_cross_attention, cross_attention_norm=cross_attention_norm)
    elif up_block_type == 'AttnUpBlock2D':
        if add_upsample is False:
            upsample_type = None
        else:
            upsample_type = upsample_type or 'conv'
        return AttnUpBlock2D(num_layers=num_layers, in_channels=in_channels,
            out_channels=out_channels, prev_output_channel=
            prev_output_channel, temb_channels=temb_channels,
            resolution_idx=resolution_idx, dropout=dropout, resnet_eps=
            resnet_eps, resnet_act_fn=resnet_act_fn, resnet_groups=
            resnet_groups, attention_head_dim=attention_head_dim,
            resnet_time_scale_shift=resnet_time_scale_shift, upsample_type=
            upsample_type)
    elif up_block_type == 'SkipUpBlock2D':
        return SkipUpBlock2D(num_layers=num_layers, in_channels=in_channels,
            out_channels=out_channels, prev_output_channel=
            prev_output_channel, temb_channels=temb_channels,
            resolution_idx=resolution_idx, dropout=dropout, add_upsample=
            add_upsample, resnet_eps=resnet_eps, resnet_act_fn=
            resnet_act_fn, resnet_time_scale_shift=resnet_time_scale_shift)
    elif up_block_type == 'AttnSkipUpBlock2D':
        return AttnSkipUpBlock2D(num_layers=num_layers, in_channels=
            in_channels, out_channels=out_channels, prev_output_channel=
            prev_output_channel, temb_channels=temb_channels,
            resolution_idx=resolution_idx, dropout=dropout, add_upsample=
            add_upsample, resnet_eps=resnet_eps, resnet_act_fn=
            resnet_act_fn, attention_head_dim=attention_head_dim,
            resnet_time_scale_shift=resnet_time_scale_shift)
    elif up_block_type == 'UpDecoderBlock2D':
        return UpDecoderBlock2D(num_layers=num_layers, in_channels=
            in_channels, out_channels=out_channels, resolution_idx=
            resolution_idx, dropout=dropout, add_upsample=add_upsample,
            resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups, resnet_time_scale_shift=
            resnet_time_scale_shift, temb_channels=temb_channels)
    elif up_block_type == 'AttnUpDecoderBlock2D':
        return AttnUpDecoderBlock2D(num_layers=num_layers, in_channels=
            in_channels, out_channels=out_channels, resolution_idx=
            resolution_idx, dropout=dropout, add_upsample=add_upsample,
            resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups, attention_head_dim=
            attention_head_dim, resnet_time_scale_shift=
            resnet_time_scale_shift, temb_channels=temb_channels)
    elif up_block_type == 'KUpBlock2D':
        return KUpBlock2D(num_layers=num_layers, in_channels=in_channels,
            out_channels=out_channels, temb_channels=temb_channels,
            resolution_idx=resolution_idx, dropout=dropout, add_upsample=
            add_upsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn)
    elif up_block_type == 'KCrossAttnUpBlock2D':
        return KCrossAttnUpBlock2D(num_layers=num_layers, in_channels=
            in_channels, out_channels=out_channels, temb_channels=
            temb_channels, resolution_idx=resolution_idx, dropout=dropout,
            add_upsample=add_upsample, resnet_eps=resnet_eps, resnet_act_fn
            =resnet_act_fn, cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim)
    raise ValueError(f'{up_block_type} does not exist.')
