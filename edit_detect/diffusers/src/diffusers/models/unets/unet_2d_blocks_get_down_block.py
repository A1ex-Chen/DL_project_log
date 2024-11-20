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
    if attention_head_dim is None:
        logger.warning(
            f'It is recommended to provide `attention_head_dim` when calling `get_down_block`. Defaulting `attention_head_dim` to {num_attention_heads}.'
            )
        attention_head_dim = num_attention_heads
    down_block_type = down_block_type[7:] if down_block_type.startswith(
        'UNetRes') else down_block_type
    if down_block_type == 'DownBlock2D':
        return DownBlock2D(num_layers=num_layers, in_channels=in_channels,
            out_channels=out_channels, temb_channels=temb_channels, dropout
            =dropout, add_downsample=add_downsample, resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups,
            downsample_padding=downsample_padding, resnet_time_scale_shift=
            resnet_time_scale_shift)
    elif down_block_type == 'ResnetDownsampleBlock2D':
        return ResnetDownsampleBlock2D(num_layers=num_layers, in_channels=
            in_channels, out_channels=out_channels, temb_channels=
            temb_channels, dropout=dropout, add_downsample=add_downsample,
            resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups, resnet_time_scale_shift=
            resnet_time_scale_shift, skip_time_act=resnet_skip_time_act,
            output_scale_factor=resnet_out_scale_factor)
    elif down_block_type == 'AttnDownBlock2D':
        if add_downsample is False:
            downsample_type = None
        else:
            downsample_type = downsample_type or 'conv'
        return AttnDownBlock2D(num_layers=num_layers, in_channels=
            in_channels, out_channels=out_channels, temb_channels=
            temb_channels, dropout=dropout, resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups,
            downsample_padding=downsample_padding, attention_head_dim=
            attention_head_dim, resnet_time_scale_shift=
            resnet_time_scale_shift, downsample_type=downsample_type)
    elif down_block_type == 'CrossAttnDownBlock2D':
        if cross_attention_dim is None:
            raise ValueError(
                'cross_attention_dim must be specified for CrossAttnDownBlock2D'
                )
        return CrossAttnDownBlock2D(num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels, out_channels=out_channels,
            temb_channels=temb_channels, dropout=dropout, add_downsample=
            add_downsample, resnet_eps=resnet_eps, resnet_act_fn=
            resnet_act_fn, resnet_groups=resnet_groups, downsample_padding=
            downsample_padding, cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads, dual_cross_attention=
            dual_cross_attention, use_linear_projection=
            use_linear_projection, only_cross_attention=
            only_cross_attention, upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift, attention_type
            =attention_type)
    elif down_block_type == 'SimpleCrossAttnDownBlock2D':
        if cross_attention_dim is None:
            raise ValueError(
                'cross_attention_dim must be specified for SimpleCrossAttnDownBlock2D'
                )
        return SimpleCrossAttnDownBlock2D(num_layers=num_layers,
            in_channels=in_channels, out_channels=out_channels,
            temb_channels=temb_channels, dropout=dropout, add_downsample=
            add_downsample, resnet_eps=resnet_eps, resnet_act_fn=
            resnet_act_fn, resnet_groups=resnet_groups, cross_attention_dim
            =cross_attention_dim, attention_head_dim=attention_head_dim,
            resnet_time_scale_shift=resnet_time_scale_shift, skip_time_act=
            resnet_skip_time_act, output_scale_factor=
            resnet_out_scale_factor, only_cross_attention=
            only_cross_attention, cross_attention_norm=cross_attention_norm)
    elif down_block_type == 'SkipDownBlock2D':
        return SkipDownBlock2D(num_layers=num_layers, in_channels=
            in_channels, out_channels=out_channels, temb_channels=
            temb_channels, dropout=dropout, add_downsample=add_downsample,
            resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn,
            downsample_padding=downsample_padding, resnet_time_scale_shift=
            resnet_time_scale_shift)
    elif down_block_type == 'AttnSkipDownBlock2D':
        return AttnSkipDownBlock2D(num_layers=num_layers, in_channels=
            in_channels, out_channels=out_channels, temb_channels=
            temb_channels, dropout=dropout, add_downsample=add_downsample,
            resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn,
            attention_head_dim=attention_head_dim, resnet_time_scale_shift=
            resnet_time_scale_shift)
    elif down_block_type == 'DownEncoderBlock2D':
        return DownEncoderBlock2D(num_layers=num_layers, in_channels=
            in_channels, out_channels=out_channels, dropout=dropout,
            add_downsample=add_downsample, resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups,
            downsample_padding=downsample_padding, resnet_time_scale_shift=
            resnet_time_scale_shift)
    elif down_block_type == 'AttnDownEncoderBlock2D':
        return AttnDownEncoderBlock2D(num_layers=num_layers, in_channels=
            in_channels, out_channels=out_channels, dropout=dropout,
            add_downsample=add_downsample, resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups,
            downsample_padding=downsample_padding, attention_head_dim=
            attention_head_dim, resnet_time_scale_shift=resnet_time_scale_shift
            )
    elif down_block_type == 'KDownBlock2D':
        return KDownBlock2D(num_layers=num_layers, in_channels=in_channels,
            out_channels=out_channels, temb_channels=temb_channels, dropout
            =dropout, add_downsample=add_downsample, resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn)
    elif down_block_type == 'KCrossAttnDownBlock2D':
        return KCrossAttnDownBlock2D(num_layers=num_layers, in_channels=
            in_channels, out_channels=out_channels, temb_channels=
            temb_channels, dropout=dropout, add_downsample=add_downsample,
            resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn,
            cross_attention_dim=cross_attention_dim, attention_head_dim=
            attention_head_dim, add_self_attention=True if not
            add_downsample else False)
    raise ValueError(f'{down_block_type} does not exist.')
