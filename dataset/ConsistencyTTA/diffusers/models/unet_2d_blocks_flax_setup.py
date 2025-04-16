def setup(self):
    resnets = [FlaxResnetBlock2D(in_channels=self.in_channels, out_channels
        =self.in_channels, dropout_prob=self.dropout, dtype=self.dtype)]
    attentions = []
    for _ in range(self.num_layers):
        attn_block = FlaxTransformer2DModel(in_channels=self.in_channels,
            n_heads=self.attn_num_head_channels, d_head=self.in_channels //
            self.attn_num_head_channels, depth=1, use_linear_projection=
            self.use_linear_projection, dtype=self.dtype)
        attentions.append(attn_block)
        res_block = FlaxResnetBlock2D(in_channels=self.in_channels,
            out_channels=self.in_channels, dropout_prob=self.dropout, dtype
            =self.dtype)
        resnets.append(res_block)
    self.resnets = resnets
    self.attentions = attentions
