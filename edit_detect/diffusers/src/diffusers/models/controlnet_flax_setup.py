def setup(self) ->None:
    block_out_channels = self.block_out_channels
    time_embed_dim = block_out_channels[0] * 4
    num_attention_heads = self.num_attention_heads or self.attention_head_dim
    self.conv_in = nn.Conv(block_out_channels[0], kernel_size=(3, 3),
        strides=(1, 1), padding=((1, 1), (1, 1)), dtype=self.dtype)
    self.time_proj = FlaxTimesteps(block_out_channels[0], flip_sin_to_cos=
        self.flip_sin_to_cos, freq_shift=self.config.freq_shift)
    self.time_embedding = FlaxTimestepEmbedding(time_embed_dim, dtype=self.
        dtype)
    self.controlnet_cond_embedding = FlaxControlNetConditioningEmbedding(
        conditioning_embedding_channels=block_out_channels[0],
        block_out_channels=self.conditioning_embedding_out_channels)
    only_cross_attention = self.only_cross_attention
    if isinstance(only_cross_attention, bool):
        only_cross_attention = (only_cross_attention,) * len(self.
            down_block_types)
    if isinstance(num_attention_heads, int):
        num_attention_heads = (num_attention_heads,) * len(self.
            down_block_types)
    down_blocks = []
    controlnet_down_blocks = []
    output_channel = block_out_channels[0]
    controlnet_block = nn.Conv(output_channel, kernel_size=(1, 1), padding=
        'VALID', kernel_init=nn.initializers.zeros_init(), bias_init=nn.
        initializers.zeros_init(), dtype=self.dtype)
    controlnet_down_blocks.append(controlnet_block)
    for i, down_block_type in enumerate(self.down_block_types):
        input_channel = output_channel
        output_channel = block_out_channels[i]
        is_final_block = i == len(block_out_channels) - 1
        if down_block_type == 'CrossAttnDownBlock2D':
            down_block = FlaxCrossAttnDownBlock2D(in_channels=input_channel,
                out_channels=output_channel, dropout=self.dropout,
                num_layers=self.layers_per_block, num_attention_heads=
                num_attention_heads[i], add_downsample=not is_final_block,
                use_linear_projection=self.use_linear_projection,
                only_cross_attention=only_cross_attention[i], dtype=self.dtype)
        else:
            down_block = FlaxDownBlock2D(in_channels=input_channel,
                out_channels=output_channel, dropout=self.dropout,
                num_layers=self.layers_per_block, add_downsample=not
                is_final_block, dtype=self.dtype)
        down_blocks.append(down_block)
        for _ in range(self.layers_per_block):
            controlnet_block = nn.Conv(output_channel, kernel_size=(1, 1),
                padding='VALID', kernel_init=nn.initializers.zeros_init(),
                bias_init=nn.initializers.zeros_init(), dtype=self.dtype)
            controlnet_down_blocks.append(controlnet_block)
        if not is_final_block:
            controlnet_block = nn.Conv(output_channel, kernel_size=(1, 1),
                padding='VALID', kernel_init=nn.initializers.zeros_init(),
                bias_init=nn.initializers.zeros_init(), dtype=self.dtype)
            controlnet_down_blocks.append(controlnet_block)
    self.down_blocks = down_blocks
    self.controlnet_down_blocks = controlnet_down_blocks
    mid_block_channel = block_out_channels[-1]
    self.mid_block = FlaxUNetMidBlock2DCrossAttn(in_channels=
        mid_block_channel, dropout=self.dropout, num_attention_heads=
        num_attention_heads[-1], use_linear_projection=self.
        use_linear_projection, dtype=self.dtype)
    self.controlnet_mid_block = nn.Conv(mid_block_channel, kernel_size=(1, 
        1), padding='VALID', kernel_init=nn.initializers.zeros_init(),
        bias_init=nn.initializers.zeros_init(), dtype=self.dtype)
