def setup(self) ->None:
    block_out_channels = self.block_out_channels
    time_embed_dim = block_out_channels[0] * 4
    if self.num_attention_heads is not None:
        raise ValueError(
            'At the moment it is not possible to define the number of attention heads via `num_attention_heads` because of a naming issue as described in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131. Passing `num_attention_heads` will only be supported in diffusers v0.19.'
            )
    num_attention_heads = self.num_attention_heads or self.attention_head_dim
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
    if isinstance(num_attention_heads, int):
        num_attention_heads = (num_attention_heads,) * len(self.
            down_block_types)
    transformer_layers_per_block = self.transformer_layers_per_block
    if isinstance(transformer_layers_per_block, int):
        transformer_layers_per_block = [transformer_layers_per_block] * len(
            self.down_block_types)
    if self.addition_embed_type is None:
        self.add_embedding = None
    elif self.addition_embed_type == 'text_time':
        if self.addition_time_embed_dim is None:
            raise ValueError(
                f'addition_embed_type {self.addition_embed_type} requires `addition_time_embed_dim` to not be None'
                )
        self.add_time_proj = FlaxTimesteps(self.addition_time_embed_dim,
            self.flip_sin_to_cos, self.freq_shift)
        self.add_embedding = FlaxTimestepEmbedding(time_embed_dim, dtype=
            self.dtype)
    else:
        raise ValueError(
            f'addition_embed_type: {self.addition_embed_type} must be None or `text_time`.'
            )
    down_blocks = []
    output_channel = block_out_channels[0]
    for i, down_block_type in enumerate(self.down_block_types):
        input_channel = output_channel
        output_channel = block_out_channels[i]
        is_final_block = i == len(block_out_channels) - 1
        if down_block_type == 'CrossAttnDownBlock2D':
            down_block = FlaxCrossAttnDownBlock2D(in_channels=input_channel,
                out_channels=output_channel, dropout=self.dropout,
                num_layers=self.layers_per_block,
                transformer_layers_per_block=transformer_layers_per_block[i
                ], num_attention_heads=num_attention_heads[i],
                add_downsample=not is_final_block, use_linear_projection=
                self.use_linear_projection, only_cross_attention=
                only_cross_attention[i], use_memory_efficient_attention=
                self.use_memory_efficient_attention, split_head_dim=self.
                split_head_dim, dtype=self.dtype)
        else:
            down_block = FlaxDownBlock2D(in_channels=input_channel,
                out_channels=output_channel, dropout=self.dropout,
                num_layers=self.layers_per_block, add_downsample=not
                is_final_block, dtype=self.dtype)
        down_blocks.append(down_block)
    self.down_blocks = down_blocks
    if self.config.mid_block_type == 'UNetMidBlock2DCrossAttn':
        self.mid_block = FlaxUNetMidBlock2DCrossAttn(in_channels=
            block_out_channels[-1], dropout=self.dropout,
            num_attention_heads=num_attention_heads[-1],
            transformer_layers_per_block=transformer_layers_per_block[-1],
            use_linear_projection=self.use_linear_projection,
            use_memory_efficient_attention=self.
            use_memory_efficient_attention, split_head_dim=self.
            split_head_dim, dtype=self.dtype)
    elif self.config.mid_block_type is None:
        self.mid_block = None
    else:
        raise ValueError(
            f'Unexpected mid_block_type {self.config.mid_block_type}')
    up_blocks = []
    reversed_block_out_channels = list(reversed(block_out_channels))
    reversed_num_attention_heads = list(reversed(num_attention_heads))
    only_cross_attention = list(reversed(only_cross_attention))
    output_channel = reversed_block_out_channels[0]
    reversed_transformer_layers_per_block = list(reversed(
        transformer_layers_per_block))
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
                transformer_layers_per_block=
                reversed_transformer_layers_per_block[i],
                num_attention_heads=reversed_num_attention_heads[i],
                add_upsample=not is_final_block, dropout=self.dropout,
                use_linear_projection=self.use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                use_memory_efficient_attention=self.
                use_memory_efficient_attention, split_head_dim=self.
                split_head_dim, dtype=self.dtype)
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
