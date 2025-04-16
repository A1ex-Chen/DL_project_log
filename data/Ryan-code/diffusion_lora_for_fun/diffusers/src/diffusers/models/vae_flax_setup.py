def setup(self):
    self.encoder = FlaxEncoder(in_channels=self.config.in_channels,
        out_channels=self.config.latent_channels, down_block_types=self.
        config.down_block_types, block_out_channels=self.config.
        block_out_channels, layers_per_block=self.config.layers_per_block,
        act_fn=self.config.act_fn, norm_num_groups=self.config.
        norm_num_groups, double_z=True, dtype=self.dtype)
    self.decoder = FlaxDecoder(in_channels=self.config.latent_channels,
        out_channels=self.config.out_channels, up_block_types=self.config.
        up_block_types, block_out_channels=self.config.block_out_channels,
        layers_per_block=self.config.layers_per_block, norm_num_groups=self
        .config.norm_num_groups, act_fn=self.config.act_fn, dtype=self.dtype)
    self.quant_conv = nn.Conv(2 * self.config.latent_channels, kernel_size=
        (1, 1), strides=(1, 1), padding='VALID', dtype=self.dtype)
    self.post_quant_conv = nn.Conv(self.config.latent_channels, kernel_size
        =(1, 1), strides=(1, 1), padding='VALID', dtype=self.dtype)
