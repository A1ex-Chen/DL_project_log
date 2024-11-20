def _init_vectorized_inputs(self, norm_type):
    assert self.config.sample_size is not None, 'Transformer2DModel over discrete input must provide sample_size'
    assert self.config.num_vector_embeds is not None, 'Transformer2DModel over discrete input must provide num_embed'
    self.height = self.config.sample_size
    self.width = self.config.sample_size
    self.num_latent_pixels = self.height * self.width
    self.latent_image_embedding = ImagePositionalEmbeddings(num_embed=self.
        config.num_vector_embeds, embed_dim=self.inner_dim, height=self.
        height, width=self.width)
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
    self.norm_out = nn.LayerNorm(self.inner_dim)
    self.out = nn.Linear(self.inner_dim, self.config.num_vector_embeds - 1)
