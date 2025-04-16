@register_to_config
def __init__(self, sample_size: Optional[int]=None, in_channels: int=4,
    out_channels: int=4, down_block_types: Tuple[str, ...]=(
    'CrossAttnDownBlockMotion', 'CrossAttnDownBlockMotion',
    'CrossAttnDownBlockMotion', 'DownBlockMotion'), up_block_types: Tuple[
    str, ...]=('UpBlockMotion', 'CrossAttnUpBlockMotion',
    'CrossAttnUpBlockMotion', 'CrossAttnUpBlockMotion'), block_out_channels:
    Tuple[int, ...]=(320, 640, 1280, 1280), layers_per_block: int=2,
    downsample_padding: int=1, mid_block_scale_factor: float=1, act_fn: str
    ='silu', norm_num_groups: int=32, norm_eps: float=1e-05,
    cross_attention_dim: int=1280, transformer_layers_per_block: Union[int,
    Tuple[int], Tuple[Tuple]]=1, reverse_transformer_layers_per_block:
    Optional[Tuple[Tuple[int]]]=None, use_linear_projection: bool=False,
    num_attention_heads: Union[int, Tuple[int, ...]]=8,
    motion_max_seq_length: int=32, motion_num_attention_heads: int=8,
    use_motion_mid_block: int=True, encoder_hid_dim: Optional[int]=None,
    encoder_hid_dim_type: Optional[str]=None, addition_embed_type: Optional
    [str]=None, addition_time_embed_dim: Optional[int]=None,
    projection_class_embeddings_input_dim: Optional[int]=None,
    time_cond_proj_dim: Optional[int]=None):
    super().__init__()
    self.sample_size = sample_size
    if len(down_block_types) != len(up_block_types):
        raise ValueError(
            f'Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}.'
            )
    if len(block_out_channels) != len(down_block_types):
        raise ValueError(
            f'Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}.'
            )
    if not isinstance(num_attention_heads, int) and len(num_attention_heads
        ) != len(down_block_types):
        raise ValueError(
            f'Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}.'
            )
    if isinstance(cross_attention_dim, list) and len(cross_attention_dim
        ) != len(down_block_types):
        raise ValueError(
            f'Must provide the same number of `cross_attention_dim` as `down_block_types`. `cross_attention_dim`: {cross_attention_dim}. `down_block_types`: {down_block_types}.'
            )
    if not isinstance(layers_per_block, int) and len(layers_per_block) != len(
        down_block_types):
        raise ValueError(
            f'Must provide the same number of `layers_per_block` as `down_block_types`. `layers_per_block`: {layers_per_block}. `down_block_types`: {down_block_types}.'
            )
    if isinstance(transformer_layers_per_block, list
        ) and reverse_transformer_layers_per_block is None:
        for layer_number_per_block in transformer_layers_per_block:
            if isinstance(layer_number_per_block, list):
                raise ValueError(
                    "Must provide 'reverse_transformer_layers_per_block` if using asymmetrical UNet."
                    )
    conv_in_kernel = 3
    conv_out_kernel = 3
    conv_in_padding = (conv_in_kernel - 1) // 2
    self.conv_in = nn.Conv2d(in_channels, block_out_channels[0],
        kernel_size=conv_in_kernel, padding=conv_in_padding)
    time_embed_dim = block_out_channels[0] * 4
    self.time_proj = Timesteps(block_out_channels[0], True, 0)
    timestep_input_dim = block_out_channels[0]
    self.time_embedding = TimestepEmbedding(timestep_input_dim,
        time_embed_dim, act_fn=act_fn, cond_proj_dim=time_cond_proj_dim)
    if encoder_hid_dim_type is None:
        self.encoder_hid_proj = None
    if addition_embed_type == 'text_time':
        self.add_time_proj = Timesteps(addition_time_embed_dim, True, 0)
        self.add_embedding = TimestepEmbedding(
            projection_class_embeddings_input_dim, time_embed_dim)
    self.down_blocks = nn.ModuleList([])
    self.up_blocks = nn.ModuleList([])
    if isinstance(num_attention_heads, int):
        num_attention_heads = (num_attention_heads,) * len(down_block_types)
    if isinstance(cross_attention_dim, int):
        cross_attention_dim = (cross_attention_dim,) * len(down_block_types)
    if isinstance(layers_per_block, int):
        layers_per_block = [layers_per_block] * len(down_block_types)
    if isinstance(transformer_layers_per_block, int):
        transformer_layers_per_block = [transformer_layers_per_block] * len(
            down_block_types)
    output_channel = block_out_channels[0]
    for i, down_block_type in enumerate(down_block_types):
        input_channel = output_channel
        output_channel = block_out_channels[i]
        is_final_block = i == len(block_out_channels) - 1
        down_block = get_down_block(down_block_type, num_layers=
            layers_per_block[i], in_channels=input_channel, out_channels=
            output_channel, temb_channels=time_embed_dim, add_downsample=
            not is_final_block, resnet_eps=norm_eps, resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups, cross_attention_dim=
            cross_attention_dim[i], num_attention_heads=num_attention_heads
            [i], downsample_padding=downsample_padding,
            use_linear_projection=use_linear_projection,
            dual_cross_attention=False, temporal_num_attention_heads=
            motion_num_attention_heads, temporal_max_seq_length=
            motion_max_seq_length, transformer_layers_per_block=
            transformer_layers_per_block[i])
        self.down_blocks.append(down_block)
    if use_motion_mid_block:
        self.mid_block = UNetMidBlockCrossAttnMotion(in_channels=
            block_out_channels[-1], temb_channels=time_embed_dim,
            resnet_eps=norm_eps, resnet_act_fn=act_fn, output_scale_factor=
            mid_block_scale_factor, cross_attention_dim=cross_attention_dim
            [-1], num_attention_heads=num_attention_heads[-1],
            resnet_groups=norm_num_groups, dual_cross_attention=False,
            use_linear_projection=use_linear_projection,
            temporal_num_attention_heads=motion_num_attention_heads,
            temporal_max_seq_length=motion_max_seq_length,
            transformer_layers_per_block=transformer_layers_per_block[-1])
    else:
        self.mid_block = UNetMidBlock2DCrossAttn(in_channels=
            block_out_channels[-1], temb_channels=time_embed_dim,
            resnet_eps=norm_eps, resnet_act_fn=act_fn, output_scale_factor=
            mid_block_scale_factor, cross_attention_dim=cross_attention_dim
            [-1], num_attention_heads=num_attention_heads[-1],
            resnet_groups=norm_num_groups, dual_cross_attention=False,
            use_linear_projection=use_linear_projection,
            transformer_layers_per_block=transformer_layers_per_block[-1])
    self.num_upsamplers = 0
    reversed_block_out_channels = list(reversed(block_out_channels))
    reversed_num_attention_heads = list(reversed(num_attention_heads))
    reversed_layers_per_block = list(reversed(layers_per_block))
    reversed_cross_attention_dim = list(reversed(cross_attention_dim))
    reversed_transformer_layers_per_block = list(reversed(
        transformer_layers_per_block))
    output_channel = reversed_block_out_channels[0]
    for i, up_block_type in enumerate(up_block_types):
        is_final_block = i == len(block_out_channels) - 1
        prev_output_channel = output_channel
        output_channel = reversed_block_out_channels[i]
        input_channel = reversed_block_out_channels[min(i + 1, len(
            block_out_channels) - 1)]
        if not is_final_block:
            add_upsample = True
            self.num_upsamplers += 1
        else:
            add_upsample = False
        up_block = get_up_block(up_block_type, num_layers=
            reversed_layers_per_block[i] + 1, in_channels=input_channel,
            out_channels=output_channel, prev_output_channel=
            prev_output_channel, temb_channels=time_embed_dim, add_upsample
            =add_upsample, resnet_eps=norm_eps, resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups, cross_attention_dim=
            reversed_cross_attention_dim[i], num_attention_heads=
            reversed_num_attention_heads[i], dual_cross_attention=False,
            resolution_idx=i, use_linear_projection=use_linear_projection,
            temporal_num_attention_heads=motion_num_attention_heads,
            temporal_max_seq_length=motion_max_seq_length,
            transformer_layers_per_block=
            reversed_transformer_layers_per_block[i])
        self.up_blocks.append(up_block)
        prev_output_channel = output_channel
    if norm_num_groups is not None:
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0
            ], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = nn.SiLU()
    else:
        self.conv_norm_out = None
        self.conv_act = None
    conv_out_padding = (conv_out_kernel - 1) // 2
    self.conv_out = nn.Conv2d(block_out_channels[0], out_channels,
        kernel_size=conv_out_kernel, padding=conv_out_padding)
