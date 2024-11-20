def get_down_block(down_block_type: str, num_layers: int, in_channels: int,
    out_channels: int, temb_channels: int, add_downsample: bool, resnet_eps:
    float, resnet_act_fn: str, num_attention_heads: int, resnet_groups:
    Optional[int]=None, cross_attention_dim: Optional[int]=None,
    downsample_padding: Optional[int]=None, dual_cross_attention: bool=
    False, use_linear_projection: bool=True, only_cross_attention: bool=
    False, upcast_attention: bool=False, resnet_time_scale_shift: str=
    'default', temporal_num_attention_heads: int=8, temporal_max_seq_length:
    int=32, transformer_layers_per_block: int=1) ->Union['DownBlock3D',
    'CrossAttnDownBlock3D', 'DownBlockMotion', 'CrossAttnDownBlockMotion',
    'DownBlockSpatioTemporal', 'CrossAttnDownBlockSpatioTemporal']:
    if down_block_type == 'DownBlock3D':
        return DownBlock3D(num_layers=num_layers, in_channels=in_channels,
            out_channels=out_channels, temb_channels=temb_channels,
            add_downsample=add_downsample, resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups,
            downsample_padding=downsample_padding, resnet_time_scale_shift=
            resnet_time_scale_shift)
    elif down_block_type == 'CrossAttnDownBlock3D':
        if cross_attention_dim is None:
            raise ValueError(
                'cross_attention_dim must be specified for CrossAttnDownBlock3D'
                )
        return CrossAttnDownBlock3D(num_layers=num_layers, in_channels=
            in_channels, out_channels=out_channels, temb_channels=
            temb_channels, add_downsample=add_downsample, resnet_eps=
            resnet_eps, resnet_act_fn=resnet_act_fn, resnet_groups=
            resnet_groups, downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim, num_attention_heads=
            num_attention_heads, dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention, upcast_attention=
            upcast_attention, resnet_time_scale_shift=resnet_time_scale_shift)
    if down_block_type == 'DownBlockMotion':
        return DownBlockMotion(num_layers=num_layers, in_channels=
            in_channels, out_channels=out_channels, temb_channels=
            temb_channels, add_downsample=add_downsample, resnet_eps=
            resnet_eps, resnet_act_fn=resnet_act_fn, resnet_groups=
            resnet_groups, downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temporal_num_attention_heads=temporal_num_attention_heads,
            temporal_max_seq_length=temporal_max_seq_length)
    elif down_block_type == 'CrossAttnDownBlockMotion':
        if cross_attention_dim is None:
            raise ValueError(
                'cross_attention_dim must be specified for CrossAttnDownBlockMotion'
                )
        return CrossAttnDownBlockMotion(num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels, out_channels=out_channels,
            temb_channels=temb_channels, add_downsample=add_downsample,
            resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups, downsample_padding=
            downsample_padding, cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads, dual_cross_attention=
            dual_cross_attention, use_linear_projection=
            use_linear_projection, only_cross_attention=
            only_cross_attention, upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temporal_num_attention_heads=temporal_num_attention_heads,
            temporal_max_seq_length=temporal_max_seq_length)
    elif down_block_type == 'DownBlockSpatioTemporal':
        return DownBlockSpatioTemporal(num_layers=num_layers, in_channels=
            in_channels, out_channels=out_channels, temb_channels=
            temb_channels, add_downsample=add_downsample)
    elif down_block_type == 'CrossAttnDownBlockSpatioTemporal':
        if cross_attention_dim is None:
            raise ValueError(
                'cross_attention_dim must be specified for CrossAttnDownBlockSpatioTemporal'
                )
        return CrossAttnDownBlockSpatioTemporal(in_channels=in_channels,
            out_channels=out_channels, temb_channels=temb_channels,
            num_layers=num_layers, transformer_layers_per_block=
            transformer_layers_per_block, add_downsample=add_downsample,
            cross_attention_dim=cross_attention_dim, num_attention_heads=
            num_attention_heads)
    raise ValueError(f'{down_block_type} does not exist.')
