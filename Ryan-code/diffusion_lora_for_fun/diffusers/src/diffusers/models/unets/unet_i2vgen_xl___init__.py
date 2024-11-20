@register_to_config
def __init__(self, sample_size: Optional[int]=None, in_channels: int=4,
    out_channels: int=4, down_block_types: Tuple[str, ...]=(
    'CrossAttnDownBlock3D', 'CrossAttnDownBlock3D', 'CrossAttnDownBlock3D',
    'DownBlock3D'), up_block_types: Tuple[str, ...]=('UpBlock3D',
    'CrossAttnUpBlock3D', 'CrossAttnUpBlock3D', 'CrossAttnUpBlock3D'),
    block_out_channels: Tuple[int, ...]=(320, 640, 1280, 1280),
    layers_per_block: int=2, norm_num_groups: Optional[int]=32,
    cross_attention_dim: int=1024, attention_head_dim: Union[int, Tuple[int
    ]]=64, num_attention_heads: Optional[Union[int, Tuple[int]]]=None):
    super().__init__()
    num_attention_heads = attention_head_dim
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
    self.conv_in = nn.Conv2d(in_channels + in_channels, block_out_channels[
        0], kernel_size=3, padding=1)
    self.transformer_in = TransformerTemporalModel(num_attention_heads=8,
        attention_head_dim=num_attention_heads, in_channels=
        block_out_channels[0], num_layers=1, norm_num_groups=norm_num_groups)
    self.image_latents_proj_in = nn.Sequential(nn.Conv2d(4, in_channels * 4,
        3, padding=1), nn.SiLU(), nn.Conv2d(in_channels * 4, in_channels * 
        4, 3, stride=1, padding=1), nn.SiLU(), nn.Conv2d(in_channels * 4,
        in_channels, 3, stride=1, padding=1))
    self.image_latents_temporal_encoder = I2VGenXLTransformerTemporalEncoder(
        dim=in_channels, num_attention_heads=2, ff_inner_dim=in_channels * 
        4, attention_head_dim=in_channels, activation_fn='gelu')
    self.image_latents_context_embedding = nn.Sequential(nn.Conv2d(4, 
        in_channels * 8, 3, padding=1), nn.SiLU(), nn.AdaptiveAvgPool2d((32,
        32)), nn.Conv2d(in_channels * 8, in_channels * 16, 3, stride=2,
        padding=1), nn.SiLU(), nn.Conv2d(in_channels * 16,
        cross_attention_dim, 3, stride=2, padding=1))
    time_embed_dim = block_out_channels[0] * 4
    self.time_proj = Timesteps(block_out_channels[0], True, 0)
    timestep_input_dim = block_out_channels[0]
    self.time_embedding = TimestepEmbedding(timestep_input_dim,
        time_embed_dim, act_fn='silu')
    self.context_embedding = nn.Sequential(nn.Linear(cross_attention_dim,
        time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, 
        cross_attention_dim * in_channels))
    self.fps_embedding = nn.Sequential(nn.Linear(timestep_input_dim,
        time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim))
    self.down_blocks = nn.ModuleList([])
    self.up_blocks = nn.ModuleList([])
    if isinstance(num_attention_heads, int):
        num_attention_heads = (num_attention_heads,) * len(down_block_types)
    output_channel = block_out_channels[0]
    for i, down_block_type in enumerate(down_block_types):
        input_channel = output_channel
        output_channel = block_out_channels[i]
        is_final_block = i == len(block_out_channels) - 1
        down_block = get_down_block(down_block_type, num_layers=
            layers_per_block, in_channels=input_channel, out_channels=
            output_channel, temb_channels=time_embed_dim, add_downsample=
            not is_final_block, resnet_eps=1e-05, resnet_act_fn='silu',
            resnet_groups=norm_num_groups, cross_attention_dim=
            cross_attention_dim, num_attention_heads=num_attention_heads[i],
            downsample_padding=1, dual_cross_attention=False)
        self.down_blocks.append(down_block)
    self.mid_block = UNetMidBlock3DCrossAttn(in_channels=block_out_channels
        [-1], temb_channels=time_embed_dim, resnet_eps=1e-05, resnet_act_fn
        ='silu', output_scale_factor=1, cross_attention_dim=
        cross_attention_dim, num_attention_heads=num_attention_heads[-1],
        resnet_groups=norm_num_groups, dual_cross_attention=False)
    self.num_upsamplers = 0
    reversed_block_out_channels = list(reversed(block_out_channels))
    reversed_num_attention_heads = list(reversed(num_attention_heads))
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
        up_block = get_up_block(up_block_type, num_layers=layers_per_block +
            1, in_channels=input_channel, out_channels=output_channel,
            prev_output_channel=prev_output_channel, temb_channels=
            time_embed_dim, add_upsample=add_upsample, resnet_eps=1e-05,
            resnet_act_fn='silu', resnet_groups=norm_num_groups,
            cross_attention_dim=cross_attention_dim, num_attention_heads=
            reversed_num_attention_heads[i], dual_cross_attention=False,
            resolution_idx=i)
        self.up_blocks.append(up_block)
        prev_output_channel = output_channel
    self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0],
        num_groups=norm_num_groups, eps=1e-05)
    self.conv_act = get_activation('silu')
    self.conv_out = nn.Conv2d(block_out_channels[0], out_channels,
        kernel_size=3, padding=1)
