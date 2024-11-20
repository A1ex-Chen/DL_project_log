def get_up_block(up_block_type, num_layers, in_channels, out_channels,
    prev_output_channel, temb_channels, add_upsample, resnet_eps,
    resnet_act_fn, num_attention_heads=None, resnet_groups=None,
    cross_attention_dim=None, dual_cross_attention=False,
    use_linear_projection=False, only_cross_attention=False,
    upcast_attention=False, resnet_time_scale_shift='default',
    resnet_skip_time_act=False, resnet_out_scale_factor=1.0,
    cross_attention_norm=None, attention_head_dim=None):
    if attention_head_dim is None:
        logger.warn(
            f'It is recommended to provide `attention_head_dim` when calling `get_up_block`. Defaulting `attention_head_dim` to {num_attention_heads}.'
            )
        attention_head_dim = num_attention_heads
    up_block_type = up_block_type[7:] if up_block_type.startswith('UNetRes'
        ) else up_block_type
    if up_block_type == 'UpBlock2D':
        return UpBlock2D(num_layers=num_layers, in_channels=in_channels,
            out_channels=out_channels, prev_output_channel=
            prev_output_channel, temb_channels=temb_channels, add_upsample=
            add_upsample, resnet_eps=resnet_eps, resnet_act_fn=
            resnet_act_fn, resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift)
    elif up_block_type == 'ResnetUpsampleBlock2D':
        return ResnetUpsampleBlock2D(num_layers=num_layers, in_channels=
            in_channels, out_channels=out_channels, prev_output_channel=
            prev_output_channel, temb_channels=temb_channels, add_upsample=
            add_upsample, resnet_eps=resnet_eps, resnet_act_fn=
            resnet_act_fn, resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift, skip_time_act=
            resnet_skip_time_act, output_scale_factor=resnet_out_scale_factor)
    elif up_block_type == 'CrossAttnUpBlock2D':
        if cross_attention_dim is None:
            raise ValueError(
                'cross_attention_dim must be specified for CrossAttnUpBlock2D')
        return CrossAttnUpBlock2D(num_layers=num_layers, in_channels=
            in_channels, out_channels=out_channels, prev_output_channel=
            prev_output_channel, temb_channels=temb_channels, add_upsample=
            add_upsample, resnet_eps=resnet_eps, resnet_act_fn=
            resnet_act_fn, resnet_groups=resnet_groups, cross_attention_dim
            =cross_attention_dim, num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention, upcast_attention=
            upcast_attention, resnet_time_scale_shift=resnet_time_scale_shift)
    elif up_block_type == 'SimpleCrossAttnUpBlock2D':
        if cross_attention_dim is None:
            raise ValueError(
                'cross_attention_dim must be specified for SimpleCrossAttnUpBlock2D'
                )
        return SimpleCrossAttnUpBlock2D(num_layers=num_layers, in_channels=
            in_channels, out_channels=out_channels, prev_output_channel=
            prev_output_channel, temb_channels=temb_channels, add_upsample=
            add_upsample, resnet_eps=resnet_eps, resnet_act_fn=
            resnet_act_fn, resnet_groups=resnet_groups, cross_attention_dim
            =cross_attention_dim, attention_head_dim=attention_head_dim,
            resnet_time_scale_shift=resnet_time_scale_shift, skip_time_act=
            resnet_skip_time_act, output_scale_factor=
            resnet_out_scale_factor, only_cross_attention=
            only_cross_attention, cross_attention_norm=cross_attention_norm)
    elif up_block_type == 'AttnUpBlock2D':
        return AttnUpBlock2D(num_layers=num_layers, in_channels=in_channels,
            out_channels=out_channels, prev_output_channel=
            prev_output_channel, temb_channels=temb_channels, add_upsample=
            add_upsample, resnet_eps=resnet_eps, resnet_act_fn=
            resnet_act_fn, resnet_groups=resnet_groups, attention_head_dim=
            attention_head_dim, resnet_time_scale_shift=resnet_time_scale_shift
            )
    elif up_block_type == 'SkipUpBlock2D':
        return SkipUpBlock2D(num_layers=num_layers, in_channels=in_channels,
            out_channels=out_channels, prev_output_channel=
            prev_output_channel, temb_channels=temb_channels, add_upsample=
            add_upsample, resnet_eps=resnet_eps, resnet_act_fn=
            resnet_act_fn, resnet_time_scale_shift=resnet_time_scale_shift)
    elif up_block_type == 'AttnSkipUpBlock2D':
        return AttnSkipUpBlock2D(num_layers=num_layers, in_channels=
            in_channels, out_channels=out_channels, prev_output_channel=
            prev_output_channel, temb_channels=temb_channels, add_upsample=
            add_upsample, resnet_eps=resnet_eps, resnet_act_fn=
            resnet_act_fn, attention_head_dim=attention_head_dim,
            resnet_time_scale_shift=resnet_time_scale_shift)
    elif up_block_type == 'UpDecoderBlock2D':
        return UpDecoderBlock2D(num_layers=num_layers, in_channels=
            in_channels, out_channels=out_channels, add_upsample=
            add_upsample, resnet_eps=resnet_eps, resnet_act_fn=
            resnet_act_fn, resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift, temb_channels=
            temb_channels)
    elif up_block_type == 'AttnUpDecoderBlock2D':
        return AttnUpDecoderBlock2D(num_layers=num_layers, in_channels=
            in_channels, out_channels=out_channels, add_upsample=
            add_upsample, resnet_eps=resnet_eps, resnet_act_fn=
            resnet_act_fn, resnet_groups=resnet_groups, attention_head_dim=
            attention_head_dim, resnet_time_scale_shift=
            resnet_time_scale_shift, temb_channels=temb_channels)
    elif up_block_type == 'KUpBlock2D':
        return KUpBlock2D(num_layers=num_layers, in_channels=in_channels,
            out_channels=out_channels, temb_channels=temb_channels,
            add_upsample=add_upsample, resnet_eps=resnet_eps, resnet_act_fn
            =resnet_act_fn)
    elif up_block_type == 'KCrossAttnUpBlock2D':
        return KCrossAttnUpBlock2D(num_layers=num_layers, in_channels=
            in_channels, out_channels=out_channels, temb_channels=
            temb_channels, add_upsample=add_upsample, resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn, cross_attention_dim=
            cross_attention_dim, attention_head_dim=attention_head_dim)
    raise ValueError(f'{up_block_type} does not exist.')
