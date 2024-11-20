def get_up_block(up_block_type, num_layers, in_channels, out_channels,
    prev_output_channel, temb_channels, add_upsample, resnet_eps,
    resnet_act_fn, transformer_layers_per_block=1, num_attention_heads=None,
    resnet_groups=None, cross_attention_dim=None, use_linear_projection=
    False, only_cross_attention=False, upcast_attention=False,
    resnet_time_scale_shift='default'):
    up_block_type = up_block_type[7:] if up_block_type.startswith('UNetRes'
        ) else up_block_type
    if up_block_type == 'UpBlock2D':
        return UpBlock2D(num_layers=num_layers, in_channels=in_channels,
            out_channels=out_channels, prev_output_channel=
            prev_output_channel, temb_channels=temb_channels, add_upsample=
            add_upsample, resnet_eps=resnet_eps, resnet_act_fn=
            resnet_act_fn, resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift)
    elif up_block_type == 'CrossAttnUpBlock2D':
        if cross_attention_dim is None:
            raise ValueError(
                'cross_attention_dim must be specified for CrossAttnUpBlock2D')
        return CrossAttnUpBlock2D(num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels, out_channels=out_channels,
            prev_output_channel=prev_output_channel, temb_channels=
            temb_channels, add_upsample=add_upsample, resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim, num_attention_heads=
            num_attention_heads, use_linear_projection=
            use_linear_projection, only_cross_attention=
            only_cross_attention, upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift)
    raise ValueError(f'{up_block_type} does not exist.')
