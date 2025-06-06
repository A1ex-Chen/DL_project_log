@register_to_config
def __init__(self, sample_size: Optional[int]=None, in_channels: int=8,
    out_channels: int=4, down_block_types: Tuple[str]=(
    'CrossAttnDownBlockSpatioTemporal', 'CrossAttnDownBlockSpatioTemporal',
    'CrossAttnDownBlockSpatioTemporal', 'DownBlockSpatioTemporal'),
    up_block_types: Tuple[str]=('UpBlockSpatioTemporal',
    'CrossAttnUpBlockSpatioTemporal', 'CrossAttnUpBlockSpatioTemporal',
    'CrossAttnUpBlockSpatioTemporal'), block_out_channels: Tuple[int]=(320,
    640, 1280, 1280), addition_time_embed_dim: int=256,
    projection_class_embeddings_input_dim: int=768, layers_per_block: Union
    [int, Tuple[int]]=2, cross_attention_dim: Union[int, Tuple[int]]=1024,
    transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]]=1,
    num_attention_heads: Union[int, Tuple[int]]=(5, 10, 20, 20), num_frames:
    int=25):
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
    self.conv_in = nn.Conv2d(in_channels, block_out_channels[0],
        kernel_size=3, padding=1)
    time_embed_dim = block_out_channels[0] * 4
    self.time_proj = Timesteps(block_out_channels[0], True,
        downscale_freq_shift=0)
    timestep_input_dim = block_out_channels[0]
    self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
    self.add_time_proj = Timesteps(addition_time_embed_dim, True,
        downscale_freq_shift=0)
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
    blocks_time_embed_dim = time_embed_dim
    output_channel = block_out_channels[0]
    for i, down_block_type in enumerate(down_block_types):
        input_channel = output_channel
        output_channel = block_out_channels[i]
        is_final_block = i == len(block_out_channels) - 1
        down_block = get_down_block(down_block_type, num_layers=
            layers_per_block[i], transformer_layers_per_block=
            transformer_layers_per_block[i], in_channels=input_channel,
            out_channels=output_channel, temb_channels=
            blocks_time_embed_dim, add_downsample=not is_final_block,
            resnet_eps=1e-05, cross_attention_dim=cross_attention_dim[i],
            num_attention_heads=num_attention_heads[i], resnet_act_fn='silu')
        self.down_blocks.append(down_block)
    self.mid_block = UNetMidBlockSpatioTemporal(block_out_channels[-1],
        temb_channels=blocks_time_embed_dim, transformer_layers_per_block=
        transformer_layers_per_block[-1], cross_attention_dim=
        cross_attention_dim[-1], num_attention_heads=num_attention_heads[-1])
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
            reversed_layers_per_block[i] + 1, transformer_layers_per_block=
            reversed_transformer_layers_per_block[i], in_channels=
            input_channel, out_channels=output_channel, prev_output_channel
            =prev_output_channel, temb_channels=blocks_time_embed_dim,
            add_upsample=add_upsample, resnet_eps=1e-05, resolution_idx=i,
            cross_attention_dim=reversed_cross_attention_dim[i],
            num_attention_heads=reversed_num_attention_heads[i],
            resnet_act_fn='silu')
        self.up_blocks.append(up_block)
        prev_output_channel = output_channel
    self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0],
        num_groups=32, eps=1e-05)
    self.conv_act = nn.SiLU()
    self.conv_out = nn.Conv2d(block_out_channels[0], out_channels,
        kernel_size=3, padding=1)
