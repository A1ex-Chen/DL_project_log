def get_up_block(up_block_type: str, num_layers: int, in_channels: int,
    out_channels: int, prev_output_channel: int, temb_channels: int,
    add_upsample: bool, resnet_eps: float, resnet_act_fn: str,
    num_attention_heads: int, resolution_idx: Optional[int]=None,
    resnet_groups: Optional[int]=None, cross_attention_dim: Optional[int]=
    None, dual_cross_attention: bool=False, use_linear_projection: bool=
    True, only_cross_attention: bool=False, upcast_attention: bool=False,
    resnet_time_scale_shift: str='default', temporal_num_attention_heads:
    int=8, temporal_cross_attention_dim: Optional[int]=None,
    temporal_max_seq_length: int=32, transformer_layers_per_block: int=1,
    dropout: float=0.0) ->Union['UpBlock3D', 'CrossAttnUpBlock3D',
    'UpBlockMotion', 'CrossAttnUpBlockMotion', 'UpBlockSpatioTemporal',
    'CrossAttnUpBlockSpatioTemporal']:
    if up_block_type == 'UpBlock3D':
        return UpBlock3D(num_layers=num_layers, in_channels=in_channels,
            out_channels=out_channels, prev_output_channel=
            prev_output_channel, temb_channels=temb_channels, add_upsample=
            add_upsample, resnet_eps=resnet_eps, resnet_act_fn=
            resnet_act_fn, resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift, resolution_idx
            =resolution_idx)
    elif up_block_type == 'CrossAttnUpBlock3D':
        if cross_attention_dim is None:
            raise ValueError(
                'cross_attention_dim must be specified for CrossAttnUpBlock3D')
        return CrossAttnUpBlock3D(num_layers=num_layers, in_channels=
            in_channels, out_channels=out_channels, prev_output_channel=
            prev_output_channel, temb_channels=temb_channels, add_upsample=
            add_upsample, resnet_eps=resnet_eps, resnet_act_fn=
            resnet_act_fn, resnet_groups=resnet_groups, cross_attention_dim
            =cross_attention_dim, num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention, upcast_attention=
            upcast_attention, resnet_time_scale_shift=
            resnet_time_scale_shift, resolution_idx=resolution_idx)
    if up_block_type == 'UpBlockMotion':
        return UpBlockMotion(num_layers=num_layers, in_channels=in_channels,
            out_channels=out_channels, prev_output_channel=
            prev_output_channel, temb_channels=temb_channels, add_upsample=
            add_upsample, resnet_eps=resnet_eps, resnet_act_fn=
            resnet_act_fn, resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift, resolution_idx
            =resolution_idx, temporal_num_attention_heads=
            temporal_num_attention_heads, temporal_max_seq_length=
            temporal_max_seq_length)
    elif up_block_type == 'CrossAttnUpBlockMotion':
        if cross_attention_dim is None:
            raise ValueError(
                'cross_attention_dim must be specified for CrossAttnUpBlockMotion'
                )
        return CrossAttnUpBlockMotion(num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels, out_channels=out_channels,
            prev_output_channel=prev_output_channel, temb_channels=
            temb_channels, add_upsample=add_upsample, resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim, num_attention_heads=
            num_attention_heads, dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention, upcast_attention=
            upcast_attention, resnet_time_scale_shift=
            resnet_time_scale_shift, resolution_idx=resolution_idx,
            temporal_num_attention_heads=temporal_num_attention_heads,
            temporal_max_seq_length=temporal_max_seq_length)
    elif up_block_type == 'UpBlockSpatioTemporal':
        return UpBlockSpatioTemporal(num_layers=num_layers, in_channels=
            in_channels, out_channels=out_channels, prev_output_channel=
            prev_output_channel, temb_channels=temb_channels,
            resolution_idx=resolution_idx, add_upsample=add_upsample)
    elif up_block_type == 'CrossAttnUpBlockSpatioTemporal':
        if cross_attention_dim is None:
            raise ValueError(
                'cross_attention_dim must be specified for CrossAttnUpBlockSpatioTemporal'
                )
        return CrossAttnUpBlockSpatioTemporal(in_channels=in_channels,
            out_channels=out_channels, prev_output_channel=
            prev_output_channel, temb_channels=temb_channels, num_layers=
            num_layers, transformer_layers_per_block=
            transformer_layers_per_block, add_upsample=add_upsample,
            cross_attention_dim=cross_attention_dim, num_attention_heads=
            num_attention_heads, resolution_idx=resolution_idx)
    raise ValueError(f'{up_block_type} does not exist.')
