def _init_patched_inputs(self, norm_type):
    assert self.config.sample_size is not None, 'Transformer2DModel over patched input must provide sample_size'
    self.height = self.config.sample_size
    self.width = self.config.sample_size
    self.patch_size = self.config.patch_size
    interpolation_scale = (self.config.interpolation_scale if self.config.
        interpolation_scale is not None else max(self.config.sample_size //
        64, 1))
    self.pos_embed = PatchEmbed(height=self.config.sample_size, width=self.
        config.sample_size, patch_size=self.config.patch_size, in_channels=
        self.in_channels, embed_dim=self.inner_dim, interpolation_scale=
        interpolation_scale)
    self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(self.
        inner_dim, self.config.num_attention_heads, self.config.
        attention_head_dim, dropout=self.config.dropout,
        cross_attention_dim=self.config.cross_attention_dim, activation_fn=
        self.config.activation_fn, num_embeds_ada_norm=self.config.
        num_embeds_ada_norm, attention_bias=self.config.attention_bias,
        only_cross_attention=self.config.only_cross_attention,
        double_self_attention=self.config.double_self_attention,
        upcast_attention=self.config.upcast_attention, norm_type=norm_type,
        norm_elementwise_affine=self.config.norm_elementwise_affine,
        norm_eps=self.config.norm_eps, attention_type=self.config.
        attention_type) for _ in range(self.config.num_layers)])
    if self.config.norm_type != 'ada_norm_single':
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=
            False, eps=1e-06)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, self.config.patch_size *
            self.config.patch_size * self.out_channels)
    elif self.config.norm_type == 'ada_norm_single':
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=
            False, eps=1e-06)
        self.scale_shift_table = nn.Parameter(torch.randn(2, self.inner_dim
            ) / self.inner_dim ** 0.5)
        self.proj_out = nn.Linear(self.inner_dim, self.config.patch_size *
            self.config.patch_size * self.out_channels)
    self.adaln_single = None
    if self.config.norm_type == 'ada_norm_single':
        self.adaln_single = AdaLayerNormSingle(self.inner_dim,
            use_additional_conditions=self.use_additional_conditions)
    self.caption_projection = None
    if self.caption_channels is not None:
        self.caption_projection = PixArtAlphaTextProjection(in_features=
            self.caption_channels, hidden_size=self.inner_dim)
