@register_to_config
def __init__(self, text_dim: int=768, clip_img_dim: int=512,
    num_text_tokens: int=77, num_attention_heads: int=16,
    attention_head_dim: int=88, in_channels: Optional[int]=None,
    out_channels: Optional[int]=None, num_layers: int=1, dropout: float=0.0,
    norm_num_groups: int=32, cross_attention_dim: Optional[int]=None,
    attention_bias: bool=False, sample_size: Optional[int]=None,
    num_vector_embeds: Optional[int]=None, patch_size: Optional[int]=None,
    activation_fn: str='geglu', num_embeds_ada_norm: Optional[int]=None,
    use_linear_projection: bool=False, only_cross_attention: bool=False,
    upcast_attention: bool=False, norm_type: str='layer_norm', block_type:
    str='unidiffuser', pre_layer_norm: bool=False, use_timestep_embedding=
    False, norm_elementwise_affine: bool=True, use_patch_pos_embed=False,
    ff_final_dropout: bool=True, use_data_type_embedding: bool=False):
    super().__init__()
    self.inner_dim = num_attention_heads * attention_head_dim
    assert sample_size is not None, 'UniDiffuserModel over patched input must provide sample_size'
    self.sample_size = sample_size
    self.in_channels = in_channels
    self.out_channels = in_channels if out_channels is None else out_channels
    self.patch_size = patch_size
    self.num_patches = self.sample_size // patch_size * (self.sample_size //
        patch_size)
    self.vae_img_in = PatchEmbed(height=sample_size, width=sample_size,
        patch_size=patch_size, in_channels=in_channels, embed_dim=self.
        inner_dim, use_pos_embed=use_patch_pos_embed)
    self.clip_img_in = nn.Linear(clip_img_dim, self.inner_dim)
    self.text_in = nn.Linear(text_dim, self.inner_dim)
    self.timestep_img_proj = Timesteps(self.inner_dim, flip_sin_to_cos=True,
        downscale_freq_shift=0)
    self.timestep_img_embed = TimestepEmbedding(self.inner_dim, 4 * self.
        inner_dim, out_dim=self.inner_dim
        ) if use_timestep_embedding else nn.Identity()
    self.timestep_text_proj = Timesteps(self.inner_dim, flip_sin_to_cos=
        True, downscale_freq_shift=0)
    self.timestep_text_embed = TimestepEmbedding(self.inner_dim, 4 * self.
        inner_dim, out_dim=self.inner_dim
        ) if use_timestep_embedding else nn.Identity()
    self.num_text_tokens = num_text_tokens
    self.num_tokens = 1 + 1 + num_text_tokens + 1 + self.num_patches
    self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, self.
        inner_dim))
    self.pos_embed_drop = nn.Dropout(p=dropout)
    trunc_normal_(self.pos_embed, std=0.02)
    self.use_data_type_embedding = use_data_type_embedding
    if self.use_data_type_embedding:
        self.data_type_token_embedding = nn.Embedding(2, self.inner_dim)
        self.data_type_pos_embed_token = nn.Parameter(torch.zeros(1, 1,
            self.inner_dim))
    self.transformer = UTransformer2DModel(num_attention_heads=
        num_attention_heads, attention_head_dim=attention_head_dim,
        in_channels=in_channels, out_channels=out_channels, num_layers=
        num_layers, dropout=dropout, norm_num_groups=norm_num_groups,
        cross_attention_dim=cross_attention_dim, attention_bias=
        attention_bias, sample_size=sample_size, num_vector_embeds=
        num_vector_embeds, patch_size=patch_size, activation_fn=
        activation_fn, num_embeds_ada_norm=num_embeds_ada_norm,
        use_linear_projection=use_linear_projection, only_cross_attention=
        only_cross_attention, upcast_attention=upcast_attention, norm_type=
        norm_type, block_type=block_type, pre_layer_norm=pre_layer_norm,
        norm_elementwise_affine=norm_elementwise_affine,
        use_patch_pos_embed=use_patch_pos_embed, ff_final_dropout=
        ff_final_dropout)
    patch_dim = patch_size ** 2 * out_channels
    self.vae_img_out = nn.Linear(self.inner_dim, patch_dim)
    self.clip_img_out = nn.Linear(self.inner_dim, clip_img_dim)
    self.text_out = nn.Linear(self.inner_dim, text_dim)
