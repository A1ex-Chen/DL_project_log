@register_to_config
def __init__(self, sample_size: Optional[int]=None, in_channels: int=4,
    out_channels: int=4, down_block_types: Tuple[str]=(
    'CrossAttnDownBlock3D', 'CrossAttnDownBlock3D', 'CrossAttnDownBlock3D',
    'DownBlock3D'), up_block_types: Tuple[str]=('UpBlock3D',
    'CrossAttnUpBlock3D', 'CrossAttnUpBlock3D', 'CrossAttnUpBlock3D'),
    block_out_channels: Tuple[int]=(320, 640, 1280, 1280), layers_per_block:
    int=2, downsample_padding: int=1, mid_block_scale_factor: float=1,
    act_fn: str='silu', norm_num_groups: Optional[int]=32, norm_eps: float=
    1e-05, cross_attention_dim: int=1024, attention_head_dim: Union[int,
    Tuple[int]]=64):
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
    if not isinstance(attention_head_dim, int) and len(attention_head_dim
        ) != len(down_block_types):
        raise ValueError(
            f'Must provide the same number of `attention_head_dim` as `down_block_types`. `attention_head_dim`: {attention_head_dim}. `down_block_types`: {down_block_types}.'
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
        time_embed_dim, act_fn=act_fn)
    self.transformer_in = TransformerTemporalModel(num_attention_heads=8,
        attention_head_dim=attention_head_dim, in_channels=
        block_out_channels[0], num_layers=1)
    self.down_blocks = nn.ModuleList([])
    self.up_blocks = nn.ModuleList([])
    if isinstance(attention_head_dim, int):
        attention_head_dim = (attention_head_dim,) * len(down_block_types)
    output_channel = block_out_channels[0]
    for i, down_block_type in enumerate(down_block_types):
        input_channel = output_channel
        output_channel = block_out_channels[i]
        is_final_block = i == len(block_out_channels) - 1
        down_block = get_down_block(down_block_type, num_layers=
            layers_per_block, in_channels=input_channel, out_channels=
            output_channel, temb_channels=time_embed_dim, add_downsample=
            not is_final_block, resnet_eps=norm_eps, resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups, cross_attention_dim=
            cross_attention_dim, attn_num_head_channels=attention_head_dim[
            i], downsample_padding=downsample_padding, dual_cross_attention
            =False)
        self.down_blocks.append(down_block)
    self.mid_block = UNetMidBlock3DCrossAttn(in_channels=block_out_channels
        [-1], temb_channels=time_embed_dim, resnet_eps=norm_eps,
        resnet_act_fn=act_fn, output_scale_factor=mid_block_scale_factor,
        cross_attention_dim=cross_attention_dim, attn_num_head_channels=
        attention_head_dim[-1], resnet_groups=norm_num_groups,
        dual_cross_attention=False)
    self.num_upsamplers = 0
    reversed_block_out_channels = list(reversed(block_out_channels))
    reversed_attention_head_dim = list(reversed(attention_head_dim))
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
            time_embed_dim, add_upsample=add_upsample, resnet_eps=norm_eps,
            resnet_act_fn=act_fn, resnet_groups=norm_num_groups,
            cross_attention_dim=cross_attention_dim, attn_num_head_channels
            =reversed_attention_head_dim[i], dual_cross_attention=False)
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
