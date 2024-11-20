def _init_continuous_input(self, norm_type):
    self.norm = torch.nn.GroupNorm(num_groups=self.config.norm_num_groups,
        num_channels=self.in_channels, eps=1e-06, affine=True)
    if self.use_linear_projection:
        self.proj_in = torch.nn.Linear(self.in_channels, self.inner_dim)
    else:
        self.proj_in = torch.nn.Conv2d(self.in_channels, self.inner_dim,
            kernel_size=1, stride=1, padding=0)
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
    if self.use_linear_projection:
        self.proj_out = torch.nn.Linear(self.inner_dim, self.out_channels)
    else:
        self.proj_out = torch.nn.Conv2d(self.inner_dim, self.out_channels,
            kernel_size=1, stride=1, padding=0)
