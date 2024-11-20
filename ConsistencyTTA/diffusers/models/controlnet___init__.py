@register_to_config
def __init__(self, in_channels: int=4, flip_sin_to_cos: bool=True,
    freq_shift: int=0, down_block_types: Tuple[str]=('CrossAttnDownBlock2D',
    'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D'),
    only_cross_attention: Union[bool, Tuple[bool]]=False,
    block_out_channels: Tuple[int]=(320, 640, 1280, 1280), layers_per_block:
    int=2, downsample_padding: int=1, mid_block_scale_factor: float=1,
    act_fn: str='silu', norm_num_groups: Optional[int]=32, norm_eps: float=
    1e-05, cross_attention_dim: int=1280, attention_head_dim: Union[int,
    Tuple[int]]=8, use_linear_projection: bool=False, class_embed_type:
    Optional[str]=None, num_class_embeds: Optional[int]=None,
    upcast_attention: bool=False, resnet_time_scale_shift: str='default',
    projection_class_embeddings_input_dim: Optional[int]=None,
    controlnet_conditioning_channel_order: str='rgb',
    conditioning_embedding_out_channels: Optional[Tuple[int]]=(16, 32, 96, 256)
    ):
    super().__init__()
    if len(block_out_channels) != len(down_block_types):
        raise ValueError(
            f'Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}.'
            )
    if not isinstance(only_cross_attention, bool) and len(only_cross_attention
        ) != len(down_block_types):
        raise ValueError(
            f'Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}.'
            )
    if not isinstance(attention_head_dim, int) and len(attention_head_dim
        ) != len(down_block_types):
        raise ValueError(
            f'Must provide the same number of `attention_head_dim` as `down_block_types`. `attention_head_dim`: {attention_head_dim}. `down_block_types`: {down_block_types}.'
            )
    conv_in_kernel = 3
    conv_in_padding = (conv_in_kernel - 1) // 2
    self.conv_in = nn.Conv2d(in_channels, block_out_channels[0],
        kernel_size=conv_in_kernel, padding=conv_in_padding)
    time_embed_dim = block_out_channels[0] * 4
    self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos,
        freq_shift)
    timestep_input_dim = block_out_channels[0]
    self.time_embedding = TimestepEmbedding(timestep_input_dim,
        time_embed_dim, act_fn=act_fn)
    if class_embed_type is None and num_class_embeds is not None:
        self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
    elif class_embed_type == 'timestep':
        self.class_embedding = TimestepEmbedding(timestep_input_dim,
            time_embed_dim)
    elif class_embed_type == 'identity':
        self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
    elif class_embed_type == 'projection':
        if projection_class_embeddings_input_dim is None:
            raise ValueError(
                "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
                )
        self.class_embedding = TimestepEmbedding(
            projection_class_embeddings_input_dim, time_embed_dim)
    else:
        self.class_embedding = None
    self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
        conditioning_embedding_channels=block_out_channels[0],
        block_out_channels=conditioning_embedding_out_channels)
    self.down_blocks = nn.ModuleList([])
    self.controlnet_down_blocks = nn.ModuleList([])
    if isinstance(only_cross_attention, bool):
        only_cross_attention = [only_cross_attention] * len(down_block_types)
    if isinstance(attention_head_dim, int):
        attention_head_dim = (attention_head_dim,) * len(down_block_types)
    output_channel = block_out_channels[0]
    controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
    controlnet_block = zero_module(controlnet_block)
    self.controlnet_down_blocks.append(controlnet_block)
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
            i], downsample_padding=downsample_padding,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention[i], upcast_attention=
            upcast_attention, resnet_time_scale_shift=resnet_time_scale_shift)
        self.down_blocks.append(down_block)
        for _ in range(layers_per_block):
            controlnet_block = nn.Conv2d(output_channel, output_channel,
                kernel_size=1)
            controlnet_block = zero_module(controlnet_block)
            self.controlnet_down_blocks.append(controlnet_block)
        if not is_final_block:
            controlnet_block = nn.Conv2d(output_channel, output_channel,
                kernel_size=1)
            controlnet_block = zero_module(controlnet_block)
            self.controlnet_down_blocks.append(controlnet_block)
    mid_block_channel = block_out_channels[-1]
    controlnet_block = nn.Conv2d(mid_block_channel, mid_block_channel,
        kernel_size=1)
    controlnet_block = zero_module(controlnet_block)
    self.controlnet_mid_block = controlnet_block
    self.mid_block = UNetMidBlock2DCrossAttn(in_channels=mid_block_channel,
        temb_channels=time_embed_dim, resnet_eps=norm_eps, resnet_act_fn=
        act_fn, output_scale_factor=mid_block_scale_factor,
        resnet_time_scale_shift=resnet_time_scale_shift,
        cross_attention_dim=cross_attention_dim, attn_num_head_channels=
        attention_head_dim[-1], resnet_groups=norm_num_groups,
        use_linear_projection=use_linear_projection, upcast_attention=
        upcast_attention)
