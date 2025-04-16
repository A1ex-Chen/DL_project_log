def setup(self):
    block_out_channels = self.block_out_channels
    time_embed_dim = block_out_channels[0] * 4
    self.conv_in = nn.Conv(block_out_channels[0], kernel_size=(3, 3),
        strides=(1, 1), padding=((1, 1), (1, 1)), dtype=self.dtype)
    self.time_proj = FlaxTimesteps(block_out_channels[0], flip_sin_to_cos=
        self.flip_sin_to_cos, freq_shift=self.config.freq_shift)
    self.time_embedding = FlaxTimestepEmbedding(time_embed_dim, dtype=self.
        dtype)
    only_cross_attention = self.only_cross_attention
    if isinstance(only_cross_attention, bool):
        only_cross_attention = (only_cross_attention,) * len(self.
            down_block_types)
    attention_head_dim = self.attention_head_dim
    if isinstance(attention_head_dim, int):
        attention_head_dim = (attention_head_dim,) * len(self.down_block_types)
    down_blocks = []
    output_channel = block_out_channels[0]
    for i, down_block_type in enumerate(self.down_block_types):
        input_channel = output_channel
        output_channel = block_out_channels[i]
        is_final_block = i == len(block_out_channels) - 1
        if down_block_type == 'CrossAttnDownBlock2D':
            down_block = FlaxCrossAttnDownBlock2D(in_channels=input_channel,
                out_channels=output_channel, dropout=self.dropout,
                num_layers=self.layers_per_block, attn_num_head_channels=
                attention_head_dim[i], add_downsample=not is_final_block,
                use_linear_projection=self.use_linear_projection,
                only_cross_attention=only_cross_attention[i], dtype=self.dtype)
        else:
            down_block = FlaxDownBlock2D(in_channels=input_channel,
                out_channels=output_channel, dropout=self.dropout,
                num_layers=self.layers_per_block, add_downsample=not
                is_final_block, dtype=self.dtype)
        down_blocks.append(down_block)
    self.down_blocks = down_blocks
    self.mid_block = FlaxUNetMidBlock2DCrossAttn(in_channels=
        block_out_channels[-1], dropout=self.dropout,
        attn_num_head_channels=attention_head_dim[-1],
        use_linear_projection=self.use_linear_projection, dtype=self.dtype)
    up_blocks = []
    reversed_block_out_channels = list(reversed(block_out_channels))
    reversed_attention_head_dim = list(reversed(attention_head_dim))
    only_cross_attention = list(reversed(only_cross_attention))
    output_channel = reversed_block_out_channels[0]
    for i, up_block_type in enumerate(self.up_block_types):
        prev_output_channel = output_channel
        output_channel = reversed_block_out_channels[i]
        input_channel = reversed_block_out_channels[min(i + 1, len(
            block_out_channels) - 1)]
        is_final_block = i == len(block_out_channels) - 1
        if up_block_type == 'CrossAttnUpBlock2D':
            up_block = FlaxCrossAttnUpBlock2D(in_channels=input_channel,
                out_channels=output_channel, prev_output_channel=
                prev_output_channel, num_layers=self.layers_per_block + 1,
                attn_num_head_channels=reversed_attention_head_dim[i],
                add_upsample=not is_final_block, dropout=self.dropout,
                use_linear_projection=self.use_linear_projection,
                only_cross_attention=only_cross_attention[i], dtype=self.dtype)
        else:
            up_block = FlaxUpBlock2D(in_channels=input_channel,
                out_channels=output_channel, prev_output_channel=
                prev_output_channel, num_layers=self.layers_per_block + 1,
                add_upsample=not is_final_block, dropout=self.dropout,
                dtype=self.dtype)
        up_blocks.append(up_block)
        prev_output_channel = output_channel
    self.up_blocks = up_blocks
    self.conv_norm_out = nn.GroupNorm(num_groups=32, epsilon=1e-05)
    self.conv_out = nn.Conv(self.out_channels, kernel_size=(3, 3), strides=
        (1, 1), padding=((1, 1), (1, 1)), dtype=self.dtype)
