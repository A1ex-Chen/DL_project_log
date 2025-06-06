def get_down_block(down_block_type, num_layers, in_channels, out_channels,
    temb_channels, add_downsample, resnet_eps, resnet_act_fn,
    attn_num_head_channels, resnet_groups=None, cross_attention_dim=None,
    downsample_padding=None, dual_cross_attention=False,
    use_linear_projection=True, only_cross_attention=False,
    upcast_attention=False, resnet_time_scale_shift='default'):
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
            cross_attention_dim=cross_attention_dim, attn_num_head_channels
            =attn_num_head_channels, dual_cross_attention=
            dual_cross_attention, use_linear_projection=
            use_linear_projection, only_cross_attention=
            only_cross_attention, upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift)
    raise ValueError(f'{down_block_type} does not exist.')
