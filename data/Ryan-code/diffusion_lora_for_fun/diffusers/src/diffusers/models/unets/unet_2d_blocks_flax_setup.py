def setup(self):
    resnets = [FlaxResnetBlock2D(in_channels=self.in_channels, out_channels
        =self.in_channels, dropout_prob=self.dropout, dtype=self.dtype)]
    attentions = []
    for _ in range(self.num_layers):
        attn_block = FlaxTransformer2DModel(in_channels=self.in_channels,
            n_heads=self.num_attention_heads, d_head=self.in_channels //
            self.num_attention_heads, depth=self.
            transformer_layers_per_block, use_linear_projection=self.
            use_linear_projection, use_memory_efficient_attention=self.
            use_memory_efficient_attention, split_head_dim=self.
            split_head_dim, dtype=self.dtype)
        attentions.append(attn_block)
        res_block = FlaxResnetBlock2D(in_channels=self.in_channels,
            out_channels=self.in_channels, dropout_prob=self.dropout, dtype
            =self.dtype)
        resnets.append(res_block)
    self.resnets = resnets
    self.attentions = attentions
