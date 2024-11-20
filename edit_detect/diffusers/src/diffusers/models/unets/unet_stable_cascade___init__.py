@register_to_config
def __init__(self, in_channels: int=16, out_channels: int=16,
    timestep_ratio_embedding_dim: int=64, patch_size: int=1,
    conditioning_dim: int=2048, block_out_channels: Tuple[int]=(2048, 2048),
    num_attention_heads: Tuple[int]=(32, 32), down_num_layers_per_block:
    Tuple[int]=(8, 24), up_num_layers_per_block: Tuple[int]=(24, 8),
    down_blocks_repeat_mappers: Optional[Tuple[int]]=(1, 1),
    up_blocks_repeat_mappers: Optional[Tuple[int]]=(1, 1),
    block_types_per_layer: Tuple[Tuple[str]]=(('SDCascadeResBlock',
    'SDCascadeTimestepBlock', 'SDCascadeAttnBlock'), ('SDCascadeResBlock',
    'SDCascadeTimestepBlock', 'SDCascadeAttnBlock')), clip_text_in_channels:
    Optional[int]=None, clip_text_pooled_in_channels=1280,
    clip_image_in_channels: Optional[int]=None, clip_seq=4,
    effnet_in_channels: Optional[int]=None, pixel_mapper_in_channels:
    Optional[int]=None, kernel_size=3, dropout: Union[float, Tuple[float]]=
    (0.1, 0.1), self_attn: Union[bool, Tuple[bool]]=True,
    timestep_conditioning_type: Tuple[str]=('sca', 'crp'), switch_level:
    Optional[Tuple[bool]]=None):
    """

        Parameters:
            in_channels (`int`, defaults to 16):
                Number of channels in the input sample.
            out_channels (`int`, defaults to 16):
                Number of channels in the output sample.
            timestep_ratio_embedding_dim (`int`, defaults to 64):
                Dimension of the projected time embedding.
            patch_size (`int`, defaults to 1):
                Patch size to use for pixel unshuffling layer
            conditioning_dim (`int`, defaults to 2048):
                Dimension of the image and text conditional embedding.
            block_out_channels (Tuple[int], defaults to (2048, 2048)):
                Tuple of output channels for each block.
            num_attention_heads (Tuple[int], defaults to (32, 32)):
                Number of attention heads in each attention block. Set to -1 to if block types in a layer do not have
                attention.
            down_num_layers_per_block (Tuple[int], defaults to [8, 24]):
                Number of layers in each down block.
            up_num_layers_per_block (Tuple[int], defaults to [24, 8]):
                Number of layers in each up block.
            down_blocks_repeat_mappers (Tuple[int], optional, defaults to [1, 1]):
                Number of 1x1 Convolutional layers to repeat in each down block.
            up_blocks_repeat_mappers (Tuple[int], optional, defaults to [1, 1]):
                Number of 1x1 Convolutional layers to repeat in each up block.
            block_types_per_layer (Tuple[Tuple[str]], optional,
                defaults to (
                    ("SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"), ("SDCascadeResBlock",
                    "SDCascadeTimestepBlock", "SDCascadeAttnBlock")
                ): Block types used in each layer of the up/down blocks.
            clip_text_in_channels (`int`, *optional*, defaults to `None`):
                Number of input channels for CLIP based text conditioning.
            clip_text_pooled_in_channels (`int`, *optional*, defaults to 1280):
                Number of input channels for pooled CLIP text embeddings.
            clip_image_in_channels (`int`, *optional*):
                Number of input channels for CLIP based image conditioning.
            clip_seq (`int`, *optional*, defaults to 4):
            effnet_in_channels (`int`, *optional*, defaults to `None`):
                Number of input channels for effnet conditioning.
            pixel_mapper_in_channels (`int`, defaults to `None`):
                Number of input channels for pixel mapper conditioning.
            kernel_size (`int`, *optional*, defaults to 3):
                Kernel size to use in the block convolutional layers.
            dropout (Tuple[float], *optional*, defaults to (0.1, 0.1)):
                Dropout to use per block.
            self_attn (Union[bool, Tuple[bool]]):
                Tuple of booleans that determine whether to use self attention in a block or not.
            timestep_conditioning_type (Tuple[str], defaults to ("sca", "crp")):
                Timestep conditioning type.
            switch_level (Optional[Tuple[bool]], *optional*, defaults to `None`):
                Tuple that indicates whether upsampling or downsampling should be applied in a block
        """
    super().__init__()
    if len(block_out_channels) != len(down_num_layers_per_block):
        raise ValueError(
            f'Number of elements in `down_num_layers_per_block` must match the length of `block_out_channels`: {len(block_out_channels)}'
            )
    elif len(block_out_channels) != len(up_num_layers_per_block):
        raise ValueError(
            f'Number of elements in `up_num_layers_per_block` must match the length of `block_out_channels`: {len(block_out_channels)}'
            )
    elif len(block_out_channels) != len(down_blocks_repeat_mappers):
        raise ValueError(
            f'Number of elements in `down_blocks_repeat_mappers` must match the length of `block_out_channels`: {len(block_out_channels)}'
            )
    elif len(block_out_channels) != len(up_blocks_repeat_mappers):
        raise ValueError(
            f'Number of elements in `up_blocks_repeat_mappers` must match the length of `block_out_channels`: {len(block_out_channels)}'
            )
    elif len(block_out_channels) != len(block_types_per_layer):
        raise ValueError(
            f'Number of elements in `block_types_per_layer` must match the length of `block_out_channels`: {len(block_out_channels)}'
            )
    if isinstance(dropout, float):
        dropout = (dropout,) * len(block_out_channels)
    if isinstance(self_attn, bool):
        self_attn = (self_attn,) * len(block_out_channels)
    if effnet_in_channels is not None:
        self.effnet_mapper = nn.Sequential(nn.Conv2d(effnet_in_channels, 
            block_out_channels[0] * 4, kernel_size=1), nn.GELU(), nn.Conv2d
            (block_out_channels[0] * 4, block_out_channels[0], kernel_size=
            1), SDCascadeLayerNorm(block_out_channels[0],
            elementwise_affine=False, eps=1e-06))
    if pixel_mapper_in_channels is not None:
        self.pixels_mapper = nn.Sequential(nn.Conv2d(
            pixel_mapper_in_channels, block_out_channels[0] * 4,
            kernel_size=1), nn.GELU(), nn.Conv2d(block_out_channels[0] * 4,
            block_out_channels[0], kernel_size=1), SDCascadeLayerNorm(
            block_out_channels[0], elementwise_affine=False, eps=1e-06))
    self.clip_txt_pooled_mapper = nn.Linear(clip_text_pooled_in_channels, 
        conditioning_dim * clip_seq)
    if clip_text_in_channels is not None:
        self.clip_txt_mapper = nn.Linear(clip_text_in_channels,
            conditioning_dim)
    if clip_image_in_channels is not None:
        self.clip_img_mapper = nn.Linear(clip_image_in_channels, 
            conditioning_dim * clip_seq)
    self.clip_norm = nn.LayerNorm(conditioning_dim, elementwise_affine=
        False, eps=1e-06)
    self.embedding = nn.Sequential(nn.PixelUnshuffle(patch_size), nn.Conv2d
        (in_channels * patch_size ** 2, block_out_channels[0], kernel_size=
        1), SDCascadeLayerNorm(block_out_channels[0], elementwise_affine=
        False, eps=1e-06))

    def get_block(block_type, in_channels, nhead, c_skip=0, dropout=0,
        self_attn=True):
        if block_type == 'SDCascadeResBlock':
            return SDCascadeResBlock(in_channels, c_skip, kernel_size=
                kernel_size, dropout=dropout)
        elif block_type == 'SDCascadeAttnBlock':
            return SDCascadeAttnBlock(in_channels, conditioning_dim, nhead,
                self_attn=self_attn, dropout=dropout)
        elif block_type == 'SDCascadeTimestepBlock':
            return SDCascadeTimestepBlock(in_channels,
                timestep_ratio_embedding_dim, conds=timestep_conditioning_type)
        else:
            raise ValueError(f'Block type {block_type} not supported')
    self.down_blocks = nn.ModuleList()
    self.down_downscalers = nn.ModuleList()
    self.down_repeat_mappers = nn.ModuleList()
    for i in range(len(block_out_channels)):
        if i > 0:
            self.down_downscalers.append(nn.Sequential(SDCascadeLayerNorm(
                block_out_channels[i - 1], elementwise_affine=False, eps=
                1e-06), UpDownBlock2d(block_out_channels[i - 1],
                block_out_channels[i], mode='down', enabled=switch_level[i -
                1]) if switch_level is not None else nn.Conv2d(
                block_out_channels[i - 1], block_out_channels[i],
                kernel_size=2, stride=2)))
        else:
            self.down_downscalers.append(nn.Identity())
        down_block = nn.ModuleList()
        for _ in range(down_num_layers_per_block[i]):
            for block_type in block_types_per_layer[i]:
                block = get_block(block_type, block_out_channels[i],
                    num_attention_heads[i], dropout=dropout[i], self_attn=
                    self_attn[i])
                down_block.append(block)
        self.down_blocks.append(down_block)
        if down_blocks_repeat_mappers is not None:
            block_repeat_mappers = nn.ModuleList()
            for _ in range(down_blocks_repeat_mappers[i] - 1):
                block_repeat_mappers.append(nn.Conv2d(block_out_channels[i],
                    block_out_channels[i], kernel_size=1))
            self.down_repeat_mappers.append(block_repeat_mappers)
    self.up_blocks = nn.ModuleList()
    self.up_upscalers = nn.ModuleList()
    self.up_repeat_mappers = nn.ModuleList()
    for i in reversed(range(len(block_out_channels))):
        if i > 0:
            self.up_upscalers.append(nn.Sequential(SDCascadeLayerNorm(
                block_out_channels[i], elementwise_affine=False, eps=1e-06),
                UpDownBlock2d(block_out_channels[i], block_out_channels[i -
                1], mode='up', enabled=switch_level[i - 1]) if switch_level
                 is not None else nn.ConvTranspose2d(block_out_channels[i],
                block_out_channels[i - 1], kernel_size=2, stride=2)))
        else:
            self.up_upscalers.append(nn.Identity())
        up_block = nn.ModuleList()
        for j in range(up_num_layers_per_block[::-1][i]):
            for k, block_type in enumerate(block_types_per_layer[i]):
                c_skip = block_out_channels[i] if i < len(block_out_channels
                    ) - 1 and j == k == 0 else 0
                block = get_block(block_type, block_out_channels[i],
                    num_attention_heads[i], c_skip=c_skip, dropout=dropout[
                    i], self_attn=self_attn[i])
                up_block.append(block)
        self.up_blocks.append(up_block)
        if up_blocks_repeat_mappers is not None:
            block_repeat_mappers = nn.ModuleList()
            for _ in range(up_blocks_repeat_mappers[::-1][i] - 1):
                block_repeat_mappers.append(nn.Conv2d(block_out_channels[i],
                    block_out_channels[i], kernel_size=1))
            self.up_repeat_mappers.append(block_repeat_mappers)
    self.clf = nn.Sequential(SDCascadeLayerNorm(block_out_channels[0],
        elementwise_affine=False, eps=1e-06), nn.Conv2d(block_out_channels[
        0], out_channels * patch_size ** 2, kernel_size=1), nn.PixelShuffle
        (patch_size))
    self.gradient_checkpointing = False
