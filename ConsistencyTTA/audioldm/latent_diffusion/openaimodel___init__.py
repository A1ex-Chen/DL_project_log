def __init__(self, image_size, in_channels, model_channels, out_channels,
    num_res_blocks, attention_resolutions, dropout=0, channel_mult=(1, 2, 4,
    8), conv_resample=True, dims=2, use_checkpoint=False, use_fp16=False,
    num_heads=1, num_head_channels=-1, num_heads_upsample=-1,
    use_scale_shift_norm=False, resblock_updown=False,
    use_new_attention_order=False, pool='adaptive', *args, **kwargs):
    super().__init__()
    if num_heads_upsample == -1:
        num_heads_upsample = num_heads
    self.in_channels = in_channels
    self.model_channels = model_channels
    self.out_channels = out_channels
    self.num_res_blocks = num_res_blocks
    self.attention_resolutions = attention_resolutions
    self.dropout = dropout
    self.channel_mult = channel_mult
    self.conv_resample = conv_resample
    self.use_checkpoint = use_checkpoint
    self.dtype = th.float16 if use_fp16 else th.float32
    self.num_heads = num_heads
    self.num_head_channels = num_head_channels
    self.num_heads_upsample = num_heads_upsample
    time_embed_dim = model_channels * 4
    self.time_embed = nn.Sequential(linear(model_channels, time_embed_dim),
        nn.SiLU(), linear(time_embed_dim, time_embed_dim))
    self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims,
        in_channels, model_channels, 3, padding=1))])
    self._feature_size = model_channels
    input_block_chans = [model_channels]
    ch = model_channels
    ds = 1
    for level, mult in enumerate(channel_mult):
        for _ in range(num_res_blocks):
            layers = [ResBlock(ch, time_embed_dim, dropout, out_channels=
                mult * model_channels, dims=dims, use_checkpoint=
                use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]
            ch = mult * model_channels
            if ds in attention_resolutions:
                layers.append(AttentionBlock(ch, use_checkpoint=
                    use_checkpoint, num_heads=num_heads, num_head_channels=
                    num_head_channels, use_new_attention_order=
                    use_new_attention_order))
            self.input_blocks.append(TimestepEmbedSequential(*layers))
            self._feature_size += ch
            input_block_chans.append(ch)
        if level != len(channel_mult) - 1:
            out_ch = ch
            self.input_blocks.append(TimestepEmbedSequential(ResBlock(ch,
                time_embed_dim, dropout, out_channels=out_ch, dims=dims,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=
                use_scale_shift_norm, down=True) if resblock_updown else
                Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)))
            ch = out_ch
            input_block_chans.append(ch)
            ds *= 2
            self._feature_size += ch
    self.middle_block = TimestepEmbedSequential(ResBlock(ch, time_embed_dim,
        dropout, dims=dims, use_checkpoint=use_checkpoint,
        use_scale_shift_norm=use_scale_shift_norm), AttentionBlock(ch,
        use_checkpoint=use_checkpoint, num_heads=num_heads,
        num_head_channels=num_head_channels, use_new_attention_order=
        use_new_attention_order), ResBlock(ch, time_embed_dim, dropout,
        dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=
        use_scale_shift_norm))
    self._feature_size += ch
    self.pool = pool
    if pool == 'adaptive':
        self.out = nn.Sequential(normalization(ch), nn.SiLU(), nn.
            AdaptiveAvgPool2d((1, 1)), zero_module(conv_nd(dims, ch,
            out_channels, 1)), nn.Flatten())
    elif pool == 'attention':
        assert num_head_channels != -1
        self.out = nn.Sequential(normalization(ch), nn.SiLU(),
            AttentionPool2d(image_size // ds, ch, num_head_channels,
            out_channels))
    elif pool == 'spatial':
        self.out = nn.Sequential(nn.Linear(self._feature_size, 2048), nn.
            ReLU(), nn.Linear(2048, self.out_channels))
    elif pool == 'spatial_v2':
        self.out = nn.Sequential(nn.Linear(self._feature_size, 2048),
            normalization(2048), nn.SiLU(), nn.Linear(2048, self.out_channels))
    else:
        raise NotImplementedError(f'Unexpected {pool} pooling')
