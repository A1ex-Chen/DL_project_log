def setup(self):
    out_channels = (self.in_channels if self.out_channels is None else self
        .out_channels)
    self.norm1 = nn.GroupNorm(num_groups=32, epsilon=1e-05)
    self.conv1 = nn.Conv(out_channels, kernel_size=(3, 3), strides=(1, 1),
        padding=((1, 1), (1, 1)), dtype=self.dtype)
    self.time_emb_proj = nn.Dense(out_channels, dtype=self.dtype)
    self.norm2 = nn.GroupNorm(num_groups=32, epsilon=1e-05)
    self.dropout = nn.Dropout(self.dropout_prob)
    self.conv2 = nn.Conv(out_channels, kernel_size=(3, 3), strides=(1, 1),
        padding=((1, 1), (1, 1)), dtype=self.dtype)
    use_nin_shortcut = (self.in_channels != out_channels if self.
        use_nin_shortcut is None else self.use_nin_shortcut)
    self.conv_shortcut = None
    if use_nin_shortcut:
        self.conv_shortcut = nn.Conv(out_channels, kernel_size=(1, 1),
            strides=(1, 1), padding='VALID', dtype=self.dtype)
