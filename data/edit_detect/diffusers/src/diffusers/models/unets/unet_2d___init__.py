@register_to_config
def __init__(self, sample_size: Optional[Union[int, Tuple[int, int]]]=None,
    in_channels: int=3, out_channels: int=3, center_input_sample: bool=
    False, time_embedding_type: str='positional', freq_shift: int=0,
    flip_sin_to_cos: bool=True, down_block_types: Tuple[str, ...]=(
    'DownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D'),
    up_block_types: Tuple[str, ...]=('AttnUpBlock2D', 'AttnUpBlock2D',
    'AttnUpBlock2D', 'UpBlock2D'), block_out_channels: Tuple[int, ...]=(224,
    448, 672, 896), layers_per_block: int=2, mid_block_scale_factor: float=
    1, downsample_padding: int=1, downsample_type: str='conv',
    upsample_type: str='conv', dropout: float=0.0, act_fn: str='silu',
    attention_head_dim: Optional[int]=8, norm_num_groups: int=32,
    attn_norm_num_groups: Optional[int]=None, norm_eps: float=1e-05,
    resnet_time_scale_shift: str='default', add_attention: bool=True,
    class_embed_type: Optional[str]=None, num_class_embeds: Optional[int]=
    None, num_train_timesteps: Optional[int]=None):
    super().__init__()
    self.sample_size = sample_size
    time_embed_dim = block_out_channels[0] * 4
    if len(down_block_types) != len(up_block_types):
        raise ValueError(
            f'Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}.'
            )
    if len(block_out_channels) != len(down_block_types):
        raise ValueError(
            f'Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}.'
            )
    self.conv_in = nn.Conv2d(in_channels, block_out_channels[0],
        kernel_size=3, padding=(1, 1))
    if time_embedding_type == 'fourier':
        self.time_proj = GaussianFourierProjection(embedding_size=
            block_out_channels[0], scale=16)
        timestep_input_dim = 2 * block_out_channels[0]
    elif time_embedding_type == 'positional':
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos,
            freq_shift)
        timestep_input_dim = block_out_channels[0]
    elif time_embedding_type == 'learned':
        self.time_proj = nn.Embedding(num_train_timesteps,
            block_out_channels[0])
        timestep_input_dim = block_out_channels[0]
    self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
    if class_embed_type is None and num_class_embeds is not None:
        self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
    elif class_embed_type == 'timestep':
        self.class_embedding = TimestepEmbedding(timestep_input_dim,
            time_embed_dim)
    elif class_embed_type == 'identity':
        self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
    else:
        self.class_embedding = None
    self.down_blocks = nn.ModuleList([])
    self.mid_block = None
    self.up_blocks = nn.ModuleList([])
    output_channel = block_out_channels[0]
    for i, down_block_type in enumerate(down_block_types):
        input_channel = output_channel
        output_channel = block_out_channels[i]
        is_final_block = i == len(block_out_channels) - 1
        down_block = get_down_block(down_block_type, num_layers=
            layers_per_block, in_channels=input_channel, out_channels=
            output_channel, temb_channels=time_embed_dim, add_downsample=
            not is_final_block, resnet_eps=norm_eps, resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups, attention_head_dim=
            attention_head_dim if attention_head_dim is not None else
            output_channel, downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
            downsample_type=downsample_type, dropout=dropout)
        self.down_blocks.append(down_block)
    self.mid_block = UNetMidBlock2D(in_channels=block_out_channels[-1],
        temb_channels=time_embed_dim, dropout=dropout, resnet_eps=norm_eps,
        resnet_act_fn=act_fn, output_scale_factor=mid_block_scale_factor,
        resnet_time_scale_shift=resnet_time_scale_shift, attention_head_dim
        =attention_head_dim if attention_head_dim is not None else
        block_out_channels[-1], resnet_groups=norm_num_groups, attn_groups=
        attn_norm_num_groups, add_attention=add_attention)
    reversed_block_out_channels = list(reversed(block_out_channels))
    output_channel = reversed_block_out_channels[0]
    for i, up_block_type in enumerate(up_block_types):
        prev_output_channel = output_channel
        output_channel = reversed_block_out_channels[i]
        input_channel = reversed_block_out_channels[min(i + 1, len(
            block_out_channels) - 1)]
        is_final_block = i == len(block_out_channels) - 1
        up_block = get_up_block(up_block_type, num_layers=layers_per_block +
            1, in_channels=input_channel, out_channels=output_channel,
            prev_output_channel=prev_output_channel, temb_channels=
            time_embed_dim, add_upsample=not is_final_block, resnet_eps=
            norm_eps, resnet_act_fn=act_fn, resnet_groups=norm_num_groups,
            attention_head_dim=attention_head_dim if attention_head_dim is not
            None else output_channel, resnet_time_scale_shift=
            resnet_time_scale_shift, upsample_type=upsample_type, dropout=
            dropout)
        self.up_blocks.append(up_block)
        prev_output_channel = output_channel
    num_groups_out = norm_num_groups if norm_num_groups is not None else min(
        block_out_channels[0] // 4, 32)
    self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0],
        num_groups=num_groups_out, eps=norm_eps)
    self.conv_act = nn.SiLU()
    self.conv_out = nn.Conv2d(block_out_channels[0], out_channels,
        kernel_size=3, padding=1)
