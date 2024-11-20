@register_to_config
def __init__(self, num_attention_heads: int=16, attention_head_dim: int=88,
    in_channels: Optional[int]=None, out_channels: Optional[int]=None,
    num_layers: int=1, dropout: float=0.0, norm_num_groups: int=32,
    cross_attention_dim: Optional[int]=None, attention_bias: bool=False,
    sample_size: Optional[int]=None, num_vector_embeds: Optional[int]=None,
    patch_size: Optional[int]=None, activation_fn: str='geglu',
    num_embeds_ada_norm: Optional[int]=None, use_linear_projection: bool=
    False, only_cross_attention: bool=False, upcast_attention: bool=False,
    norm_type: str='layer_norm', norm_elementwise_affine: bool=True):
    super().__init__()
    self.use_linear_projection = use_linear_projection
    self.num_attention_heads = num_attention_heads
    self.attention_head_dim = attention_head_dim
    inner_dim = num_attention_heads * attention_head_dim
    self.is_input_continuous = in_channels is not None and patch_size is None
    self.is_input_vectorized = num_vector_embeds is not None
    self.is_input_patches = in_channels is not None and patch_size is not None
    if norm_type == 'layer_norm' and num_embeds_ada_norm is not None:
        deprecation_message = (
            f"The configuration file of this model: {self.__class__} is outdated. `norm_type` is either not set or incorrectly set to `'layer_norm'`.Make sure to set `norm_type` to `'ada_norm'` in the config. Please make sure to update the config accordingly as leaving `norm_type` might led to incorrect results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for the `transformer/config.json` file"
            )
        deprecate('norm_type!=num_embeds_ada_norm', '1.0.0',
            deprecation_message, standard_warn=False)
        norm_type = 'ada_norm'
    if self.is_input_continuous and self.is_input_vectorized:
        raise ValueError(
            f'Cannot define both `in_channels`: {in_channels} and `num_vector_embeds`: {num_vector_embeds}. Make sure that either `in_channels` or `num_vector_embeds` is None.'
            )
    elif self.is_input_vectorized and self.is_input_patches:
        raise ValueError(
            f'Cannot define both `num_vector_embeds`: {num_vector_embeds} and `patch_size`: {patch_size}. Make sure that either `num_vector_embeds` or `num_patches` is None.'
            )
    elif not self.is_input_continuous and not self.is_input_vectorized and not self.is_input_patches:
        raise ValueError(
            f'Has to define `in_channels`: {in_channels}, `num_vector_embeds`: {num_vector_embeds}, or patch_size: {patch_size}. Make sure that `in_channels`, `num_vector_embeds` or `num_patches` is not None.'
            )
    if self.is_input_continuous:
        self.in_channels = in_channels
        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups,
            num_channels=in_channels, eps=1e-06, affine=True)
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1,
                stride=1, padding=0)
    elif self.is_input_vectorized:
        assert sample_size is not None, 'Transformer2DModel over discrete input must provide sample_size'
        assert num_vector_embeds is not None, 'Transformer2DModel over discrete input must provide num_embed'
        self.height = sample_size
        self.width = sample_size
        self.num_vector_embeds = num_vector_embeds
        self.num_latent_pixels = self.height * self.width
        self.latent_image_embedding = ImagePositionalEmbeddings(num_embed=
            num_vector_embeds, embed_dim=inner_dim, height=self.height,
            width=self.width)
    elif self.is_input_patches:
        assert sample_size is not None, 'Transformer2DModel over patched input must provide sample_size'
        self.height = sample_size
        self.width = sample_size
        self.patch_size = patch_size
        self.pos_embed = PatchEmbed(height=sample_size, width=sample_size,
            patch_size=patch_size, in_channels=in_channels, embed_dim=inner_dim
            )
    self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(
        inner_dim, num_attention_heads, attention_head_dim, dropout=dropout,
        cross_attention_dim=cross_attention_dim, activation_fn=
        activation_fn, num_embeds_ada_norm=num_embeds_ada_norm,
        attention_bias=attention_bias, only_cross_attention=
        only_cross_attention, upcast_attention=upcast_attention, norm_type=
        norm_type, norm_elementwise_affine=norm_elementwise_affine) for d in
        range(num_layers)])
    self.out_channels = in_channels if out_channels is None else out_channels
    if self.is_input_continuous:
        if use_linear_projection:
            self.proj_out = nn.Linear(inner_dim, in_channels)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1,
                stride=1, padding=0)
    elif self.is_input_vectorized:
        self.norm_out = nn.LayerNorm(inner_dim)
        self.out = nn.Linear(inner_dim, self.num_vector_embeds - 1)
    elif self.is_input_patches:
        self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False,
            eps=1e-06)
        self.proj_out_1 = nn.Linear(inner_dim, 2 * inner_dim)
        self.proj_out_2 = nn.Linear(inner_dim, patch_size * patch_size *
            self.out_channels)
