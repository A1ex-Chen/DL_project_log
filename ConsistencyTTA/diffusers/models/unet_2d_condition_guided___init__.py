@register_to_config
def __init__(self, sample_size: Optional[int]=None, in_channels: int=4,
    out_channels: int=4, center_input_sample: bool=False, flip_sin_to_cos:
    bool=True, freq_shift: int=0, down_block_types: Tuple[str]=(
    'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D',
    'DownBlock2D'), mid_block_type: Optional[str]='UNetMidBlock2DCrossAttn',
    up_block_types: Tuple[str]=('UpBlock2D', 'CrossAttnUpBlock2D',
    'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D'), only_cross_attention:
    Union[bool, Tuple[bool]]=False, block_out_channels: Tuple[int]=(320, 
    640, 1280, 1280), layers_per_block: Union[int, Tuple[int]]=2,
    downsample_padding: int=1, mid_block_scale_factor: float=1, act_fn: str
    ='silu', norm_num_groups: Optional[int]=32, norm_eps: float=1e-05,
    cross_attention_dim: Union[int, Tuple[int]]=1280, encoder_hid_dim:
    Optional[int]=None, encoder_hid_dim_type: Optional[str]=None,
    attention_head_dim: Union[int, Tuple[int]]=8, num_attention_heads:
    Optional[Union[int, Tuple[int]]]=None, dual_cross_attention: bool=False,
    use_linear_projection: bool=False, class_embed_type: Optional[str]=None,
    addition_embed_type: Optional[str]=None, num_class_embeds: Optional[int
    ]=None, upcast_attention: bool=False, resnet_time_scale_shift: str=
    'default', resnet_skip_time_act: bool=False, resnet_out_scale_factor:
    int=1.0, time_embedding_type: str='positional', time_embedding_dim:
    Optional[int]=None, time_embedding_act_fn: Optional[str]=None,
    timestep_post_act: Optional[str]=None, time_cond_proj_dim: Optional[int
    ]=None, guidance_embedding_type: str='fourier', guidance_embedding_dim:
    Optional[int]=None, guidance_post_act: Optional[str]=None,
    guidance_cond_proj_dim: Optional[int]=None, conv_in_kernel: int=3,
    conv_out_kernel: int=3, projection_class_embeddings_input_dim: Optional
    [int]=None, class_embeddings_concat: bool=False,
    mid_block_only_cross_attention: Optional[bool]=None,
    cross_attention_norm: Optional[str]=None, addition_embed_type_num_heads=64
    ):
    super().__init__()
    self.sample_size = sample_size
    num_attention_heads = num_attention_heads or attention_head_dim
    if len(down_block_types) != len(up_block_types):
        raise ValueError(
            f'Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}.'
            )
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
    if not isinstance(attention_head_dim, int) and len(attention_head_dim
        ) != len(down_block_types):
        raise ValueError(
            f'Must provide the same number of `attention_head_dim` as `down_block_types`. `attention_head_dim`: {attention_head_dim}. `down_block_types`: {down_block_types}.'
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
    conv_in_padding = (conv_in_kernel - 1) // 2
    self.conv_in = nn.Conv2d(in_channels, block_out_channels[0],
        kernel_size=conv_in_kernel, padding=conv_in_padding)
    embedding_types = {'time': time_embedding_type, 'guidance':
        guidance_embedding_type}
    embedding_dims = {'time': time_embedding_dim, 'guidance':
        guidance_embedding_dim}
    embed_dims, embed_input_dims, embed_projs = {}, {}, {}
    for key in ['time', 'guidance']:
        logger.info(f'Using {embedding_types[key]} embedding for {key}.')
        if embedding_types[key] == 'fourier':
            embed_dims[key] = embedding_dims[key] or block_out_channels[0] * 4
            embed_input_dims[key] = embed_dims[key]
            if embed_dims[key] % 2 != 0:
                raise ValueError(
                    f'`{key}_embed_dim` should be divisible by 2, but is {embed_dims[key]}.'
                    )
            embed_projs[key] = GaussianFourierProjection(embed_dims[key] //
                2, set_W_to_weight=False, log=False, flip_sin_to_cos=
                flip_sin_to_cos)
        elif embedding_types[key] == 'positional':
            embed_dims[key] = embedding_dims[key] or block_out_channels[0] * 4
            embed_input_dims[key] = block_out_channels[0]
            embed_projs[key] = Timesteps(block_out_channels[0],
                flip_sin_to_cos, freq_shift)
        else:
            raise ValueError(
                f'{embedding_types[key]} does not exist for {key} embedding. Please make sure to use one of `fourier` or `positional`.'
                )
    self.time_proj, self.guidance_proj = embed_projs['time'], embed_projs[
        'guidance']
    self.time_embedding = TimestepEmbedding(embed_input_dims['time'],
        embed_dims['time'], act_fn=act_fn, post_act_fn=timestep_post_act,
        cond_proj_dim=time_cond_proj_dim)
    self.guidance_embedding = TimestepEmbedding(embed_input_dims['guidance'
        ], embed_dims['guidance'], act_fn=act_fn, post_act_fn=
        guidance_post_act, cond_proj_dim=guidance_cond_proj_dim)
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
        self.class_embedding = nn.Embedding(num_class_embeds,
            embedding_dims['time'])
    elif class_embed_type == 'timestep':
        self.class_embedding = TimestepEmbedding(embed_input_dims['time'],
            embed_dims['time'], act_fn=act_fn)
    elif class_embed_type == 'identity':
        self.class_embedding = nn.Identity(embed_dims['time'], embed_dims[
            'time'])
    elif class_embed_type == 'projection':
        if projection_class_embeddings_input_dim is None:
            raise ValueError(
                "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
                )
        self.class_embedding = TimestepEmbedding(
            projection_class_embeddings_input_dim, embed_dims['time'])
    elif class_embed_type == 'simple_projection':
        if projection_class_embeddings_input_dim is None:
            raise ValueError(
                "`class_embed_type`: 'simple_projection' requires `projection_class_embeddings_input_dim` be set"
                )
        self.class_embedding = nn.Linear(projection_class_embeddings_input_dim,
            embed_dims['time'])
    else:
        self.class_embedding = None
    if addition_embed_type == 'text':
        if encoder_hid_dim is not None:
            text_time_embedding_from_dim = encoder_hid_dim
        else:
            text_time_embedding_from_dim = cross_attention_dim
        self.add_embedding = TextTimeEmbedding(text_time_embedding_from_dim,
            embed_dims['time'], num_heads=addition_embed_type_num_heads)
    elif addition_embed_type == 'text_image':
        self.add_embedding = TextImageTimeEmbedding(text_embed_dim=
            cross_attention_dim, image_embed_dim=cross_attention_dim,
            time_embed_dim=embed_dims['time'])
    elif addition_embed_type is not None:
        raise ValueError(
            f"addition_embed_type: {addition_embed_type} must be None, 'text' or 'text_image'."
            )
    if time_embedding_act_fn is None:
        self.time_embed_act = None
    else:
        self.time_embed_act = get_activation(time_embedding_act_fn)
    self.down_blocks = nn.ModuleList([])
    self.up_blocks = nn.ModuleList([])
    if isinstance(only_cross_attention, bool):
        if mid_block_only_cross_attention is None:
            mid_block_only_cross_attention = only_cross_attention
        only_cross_attention = [only_cross_attention] * len(down_block_types)
    if mid_block_only_cross_attention is None:
        mid_block_only_cross_attention = False
    if isinstance(num_attention_heads, int):
        num_attention_heads = (num_attention_heads,) * len(down_block_types)
    if isinstance(attention_head_dim, int):
        attention_head_dim = (attention_head_dim,) * len(down_block_types)
    if isinstance(cross_attention_dim, int):
        cross_attention_dim = (cross_attention_dim,) * len(down_block_types)
    if isinstance(layers_per_block, int):
        layers_per_block = [layers_per_block] * len(down_block_types)
    if class_embeddings_concat:
        blocks_time_embed_dim = embed_dims['time'] * 3
    else:
        blocks_time_embed_dim = embed_dims['time']
    output_channel = block_out_channels[0]
    for i, down_block_type in enumerate(down_block_types):
        input_channel = output_channel
        output_channel = block_out_channels[i]
        is_final_block = i == len(block_out_channels) - 1
        down_block = get_down_block(down_block_type, num_layers=
            layers_per_block[i], in_channels=input_channel, out_channels=
            output_channel, temb_channels=blocks_time_embed_dim,
            add_downsample=not is_final_block, resnet_eps=norm_eps,
            resnet_act_fn=act_fn, resnet_groups=norm_num_groups,
            cross_attention_dim=cross_attention_dim[i], num_attention_heads
            =num_attention_heads[i], downsample_padding=downsample_padding,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention[i], upcast_attention=
            upcast_attention, resnet_time_scale_shift=
            resnet_time_scale_shift, resnet_skip_time_act=
            resnet_skip_time_act, resnet_out_scale_factor=
            resnet_out_scale_factor, cross_attention_norm=
            cross_attention_norm, attention_head_dim=attention_head_dim[i] if
            attention_head_dim[i] is not None else output_channel)
        self.down_blocks.append(down_block)
    if mid_block_type == 'UNetMidBlock2DCrossAttn':
        self.mid_block = UNetMidBlock2DCrossAttn(in_channels=
            block_out_channels[-1], temb_channels=blocks_time_embed_dim,
            resnet_eps=norm_eps, resnet_act_fn=act_fn, output_scale_factor=
            mid_block_scale_factor, resnet_time_scale_shift=
            resnet_time_scale_shift, cross_attention_dim=
            cross_attention_dim[-1], num_attention_heads=
            num_attention_heads[-1], resnet_groups=norm_num_groups,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection, upcast_attention=
            upcast_attention)
    elif mid_block_type == 'UNetMidBlock2DSimpleCrossAttn':
        self.mid_block = UNetMidBlock2DSimpleCrossAttn(in_channels=
            block_out_channels[-1], temb_channels=blocks_time_embed_dim,
            resnet_eps=norm_eps, resnet_act_fn=act_fn, output_scale_factor=
            mid_block_scale_factor, cross_attention_dim=cross_attention_dim
            [-1], attention_head_dim=attention_head_dim[-1], resnet_groups=
            norm_num_groups, resnet_time_scale_shift=
            resnet_time_scale_shift, skip_time_act=resnet_skip_time_act,
            only_cross_attention=mid_block_only_cross_attention,
            cross_attention_norm=cross_attention_norm)
    elif mid_block_type is None:
        self.mid_block = None
    else:
        raise ValueError(f'unknown mid_block_type : {mid_block_type}')
    self.num_upsamplers = 0
    reversed_block_out_channels = list(reversed(block_out_channels))
    reversed_num_attention_heads = list(reversed(num_attention_heads))
    reversed_layers_per_block = list(reversed(layers_per_block))
    reversed_cross_attention_dim = list(reversed(cross_attention_dim))
    only_cross_attention = list(reversed(only_cross_attention))
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
            prev_output_channel, temb_channels=blocks_time_embed_dim,
            add_upsample=add_upsample, resnet_eps=norm_eps, resnet_act_fn=
            act_fn, resnet_groups=norm_num_groups, cross_attention_dim=
            reversed_cross_attention_dim[i], num_attention_heads=
            reversed_num_attention_heads[i], dual_cross_attention=
            dual_cross_attention, use_linear_projection=
            use_linear_projection, only_cross_attention=
            only_cross_attention[i], upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_skip_time_act=resnet_skip_time_act,
            resnet_out_scale_factor=resnet_out_scale_factor,
            cross_attention_norm=cross_attention_norm, attention_head_dim=
            attention_head_dim[i] if attention_head_dim[i] is not None else
            output_channel)
        self.up_blocks.append(up_block)
        prev_output_channel = output_channel
    if norm_num_groups is not None:
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0
            ], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = get_activation(act_fn)
    else:
        self.conv_norm_out = None
        self.conv_act = None
    conv_out_padding = (conv_out_kernel - 1) // 2
    self.conv_out = nn.Conv2d(block_out_channels[0], out_channels,
        kernel_size=conv_out_kernel, padding=conv_out_padding)
