@register_to_config
def __init__(self, in_channels: int=4, conditioning_channels: int=3,
    flip_sin_to_cos: bool=True, freq_shift: int=0, down_block_types: Tuple[
    str, ...]=('CrossAttnDownBlock2D', 'CrossAttnDownBlock2D',
    'CrossAttnDownBlock2D', 'DownBlock2D'), mid_block_type: Optional[str]=
    'UNetMidBlock2DCrossAttn', only_cross_attention: Union[bool, Tuple[bool
    ]]=False, block_out_channels: Tuple[int, ...]=(320, 640, 1280, 1280),
    layers_per_block: int=2, downsample_padding: int=1,
    mid_block_scale_factor: float=1, act_fn: str='silu', norm_num_groups:
    Optional[int]=32, norm_eps: float=1e-05, cross_attention_dim: int=1280,
    transformer_layers_per_block: Union[int, Tuple[int, ...]]=1,
    encoder_hid_dim: Optional[int]=None, encoder_hid_dim_type: Optional[str
    ]=None, attention_head_dim: Union[int, Tuple[int, ...]]=8,
    num_attention_heads: Optional[Union[int, Tuple[int, ...]]]=None,
    use_linear_projection: bool=False, class_embed_type: Optional[str]=None,
    addition_embed_type: Optional[str]=None, addition_time_embed_dim:
    Optional[int]=None, num_class_embeds: Optional[int]=None,
    upcast_attention: bool=False, resnet_time_scale_shift: str='default',
    projection_class_embeddings_input_dim: Optional[int]=None,
    controlnet_conditioning_channel_order: str='rgb',
    conditioning_embedding_out_channels: Optional[Tuple[int, ...]]=(16, 32,
    96, 256), global_pool_conditions: bool=False,
    addition_embed_type_num_heads: int=64):
    super().__init__()
    num_attention_heads = num_attention_heads or attention_head_dim
    if len(block_out_channels) != len(down_block_types):
        raise ValueError(
            f'Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}.'
            )
    if not isinstance(only_cross_attention, bool) and len(only_cross_attention
        ) != len(down_block_types):
        raise ValueError(
            f'Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}.'
            )
    if not isinstance(num_attention_heads, int) and len(num_attention_heads
        ) != len(down_block_types):
        raise ValueError(
            f'Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}.'
            )
    if isinstance(transformer_layers_per_block, int):
        transformer_layers_per_block = [transformer_layers_per_block] * len(
            down_block_types)
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
    if encoder_hid_dim_type is None and encoder_hid_dim is not None:
        encoder_hid_dim_type = 'text_proj'
        self.register_to_config(encoder_hid_dim_type=encoder_hid_dim_type)
        logger.info(
            "encoder_hid_dim_type defaults to 'text_proj' as `encoder_hid_dim` is defined."
            )
    if encoder_hid_dim is None and encoder_hid_dim_type is not None:
        raise ValueError(
            f'`encoder_hid_dim` has to be defined when `encoder_hid_dim_type` is set to {encoder_hid_dim_type}.'
            )
    if encoder_hid_dim_type == 'text_proj':
        self.encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attention_dim)
    elif encoder_hid_dim_type == 'text_image_proj':
        self.encoder_hid_proj = TextImageProjection(text_embed_dim=
            encoder_hid_dim, image_embed_dim=cross_attention_dim,
            cross_attention_dim=cross_attention_dim)
    elif encoder_hid_dim_type is not None:
        raise ValueError(
            f"encoder_hid_dim_type: {encoder_hid_dim_type} must be None, 'text_proj' or 'text_image_proj'."
            )
    else:
        self.encoder_hid_proj = None
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
    if addition_embed_type == 'text':
        if encoder_hid_dim is not None:
            text_time_embedding_from_dim = encoder_hid_dim
        else:
            text_time_embedding_from_dim = cross_attention_dim
        self.add_embedding = TextTimeEmbedding(text_time_embedding_from_dim,
            time_embed_dim, num_heads=addition_embed_type_num_heads)
    elif addition_embed_type == 'text_image':
        self.add_embedding = TextImageTimeEmbedding(text_embed_dim=
            cross_attention_dim, image_embed_dim=cross_attention_dim,
            time_embed_dim=time_embed_dim)
    elif addition_embed_type == 'text_time':
        self.add_time_proj = Timesteps(addition_time_embed_dim,
            flip_sin_to_cos, freq_shift)
        self.add_embedding = TimestepEmbedding(
            projection_class_embeddings_input_dim, time_embed_dim)
    elif addition_embed_type is not None:
        raise ValueError(
            f"addition_embed_type: {addition_embed_type} must be None, 'text' or 'text_image'."
            )
    self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
        conditioning_embedding_channels=block_out_channels[0],
        block_out_channels=conditioning_embedding_out_channels,
        conditioning_channels=conditioning_channels)
    self.down_blocks = nn.ModuleList([])
    self.controlnet_down_blocks = nn.ModuleList([])
    if isinstance(only_cross_attention, bool):
        only_cross_attention = [only_cross_attention] * len(down_block_types)
    if isinstance(attention_head_dim, int):
        attention_head_dim = (attention_head_dim,) * len(down_block_types)
    if isinstance(num_attention_heads, int):
        num_attention_heads = (num_attention_heads,) * len(down_block_types)
    output_channel = block_out_channels[0]
    controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
    controlnet_block = zero_module(controlnet_block)
    self.controlnet_down_blocks.append(controlnet_block)
    for i, down_block_type in enumerate(down_block_types):
        input_channel = output_channel
        output_channel = block_out_channels[i]
        is_final_block = i == len(block_out_channels) - 1
        down_block = get_down_block(down_block_type, num_layers=
            layers_per_block, transformer_layers_per_block=
            transformer_layers_per_block[i], in_channels=input_channel,
            out_channels=output_channel, temb_channels=time_embed_dim,
            add_downsample=not is_final_block, resnet_eps=norm_eps,
            resnet_act_fn=act_fn, resnet_groups=norm_num_groups,
            cross_attention_dim=cross_attention_dim, num_attention_heads=
            num_attention_heads[i], attention_head_dim=attention_head_dim[i
            ] if attention_head_dim[i] is not None else output_channel,
            downsample_padding=downsample_padding, use_linear_projection=
            use_linear_projection, only_cross_attention=
            only_cross_attention[i], upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift)
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
    if mid_block_type == 'UNetMidBlock2DCrossAttn':
        self.mid_block = UNetMidBlock2DCrossAttn(transformer_layers_per_block
            =transformer_layers_per_block[-1], in_channels=
            mid_block_channel, temb_channels=time_embed_dim, resnet_eps=
            norm_eps, resnet_act_fn=act_fn, output_scale_factor=
            mid_block_scale_factor, resnet_time_scale_shift=
            resnet_time_scale_shift, cross_attention_dim=
            cross_attention_dim, num_attention_heads=num_attention_heads[-1
            ], resnet_groups=norm_num_groups, use_linear_projection=
            use_linear_projection, upcast_attention=upcast_attention)
    elif mid_block_type == 'UNetMidBlock2D':
        self.mid_block = UNetMidBlock2D(in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim, num_layers=0, resnet_eps=norm_eps,
            resnet_act_fn=act_fn, output_scale_factor=
            mid_block_scale_factor, resnet_groups=norm_num_groups,
            resnet_time_scale_shift=resnet_time_scale_shift, add_attention=
            False)
    else:
        raise ValueError(f'unknown mid_block_type : {mid_block_type}')
