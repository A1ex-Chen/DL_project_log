def get_down_block(down_block_type, num_layers, in_channels, out_channels,
    temb_channels, add_downsample, resnet_eps, resnet_act_fn,
    transformer_layers_per_block=1, num_attention_heads=None, resnet_groups
    =None, cross_attention_dim=None, downsample_padding=None,
    use_linear_projection=False, only_cross_attention=False,
    upcast_attention=False, resnet_time_scale_shift='default'):
    down_block_type = down_block_type[7:] if down_block_type.startswith(
        'UNetRes') else down_block_type
    if down_block_type == 'DownBlock2D':
        return DownBlock2D(num_layers=num_layers, in_channels=in_channels,
            out_channels=out_channels, temb_channels=temb_channels,
            add_downsample=add_downsample, resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups,
            downsample_padding=downsample_padding, resnet_time_scale_shift=
            resnet_time_scale_shift)
    elif down_block_type == 'CrossAttnDownBlock2D':
        if cross_attention_dim is None:
            raise ValueError(
                'cross_attention_dim must be specified for CrossAttnDownBlock2D'
                )
        return CrossAttnDownBlock2D(num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels, out_channels=out_channels,
            temb_channels=temb_channels, add_downsample=add_downsample,
            resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups, downsample_padding=
            downsample_padding, cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads, use_linear_projection=
            use_linear_projection, only_cross_attention=
            only_cross_attention, upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift)
    raise ValueError(f'{down_block_type} does not exist.')
