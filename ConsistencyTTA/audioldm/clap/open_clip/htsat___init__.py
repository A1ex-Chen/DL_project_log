def __init__(self, spec_size=256, patch_size=4, patch_stride=(4, 4),
    in_chans=1, num_classes=527, embed_dim=96, depths=[2, 2, 6, 2],
    num_heads=[4, 8, 16, 32], window_size=8, mlp_ratio=4.0, qkv_bias=True,
    qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
    norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=
    False, norm_before_mlp='ln', config=None, enable_fusion=False,
    fusion_type='None', **kwargs):
    super(HTSAT_Swin_Transformer, self).__init__()
    self.config = config
    self.spec_size = spec_size
    self.patch_stride = patch_stride
    self.patch_size = patch_size
    self.window_size = window_size
    self.embed_dim = embed_dim
    self.depths = depths
    self.ape = ape
    self.in_chans = in_chans
    self.num_classes = num_classes
    self.num_heads = num_heads
    self.num_layers = len(self.depths)
    self.num_features = int(self.embed_dim * 2 ** (self.num_layers - 1))
    self.drop_rate = drop_rate
    self.attn_drop_rate = attn_drop_rate
    self.drop_path_rate = drop_path_rate
    self.qkv_bias = qkv_bias
    self.qk_scale = None
    self.patch_norm = patch_norm
    self.norm_layer = norm_layer if self.patch_norm else None
    self.norm_before_mlp = norm_before_mlp
    self.mlp_ratio = mlp_ratio
    self.use_checkpoint = use_checkpoint
    self.enable_fusion = enable_fusion
    self.fusion_type = fusion_type
    self.freq_ratio = self.spec_size // self.config.mel_bins
    window = 'hann'
    center = True
    pad_mode = 'reflect'
    ref = 1.0
    amin = 1e-10
    top_db = None
    self.interpolate_ratio = 32
    self.spectrogram_extractor = Spectrogram(n_fft=config.window_size,
        hop_length=config.hop_size, win_length=config.window_size, window=
        window, center=center, pad_mode=pad_mode, freeze_parameters=True)
    self.logmel_extractor = LogmelFilterBank(sr=config.sample_rate, n_fft=
        config.window_size, n_mels=config.mel_bins, fmin=config.fmin, fmax=
        config.fmax, ref=ref, amin=amin, top_db=top_db, freeze_parameters=True)
    self.spec_augmenter = SpecAugmentation(time_drop_width=64,
        time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)
    self.bn0 = nn.BatchNorm2d(self.config.mel_bins)
    self.patch_embed = PatchEmbed(img_size=self.spec_size, patch_size=self.
        patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim,
        norm_layer=self.norm_layer, patch_stride=patch_stride,
        enable_fusion=self.enable_fusion, fusion_type=self.fusion_type)
    num_patches = self.patch_embed.num_patches
    patches_resolution = self.patch_embed.grid_size
    self.patches_resolution = patches_resolution
    if self.ape:
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches,
            self.embed_dim))
        trunc_normal_(self.absolute_pos_embed, std=0.02)
    self.pos_drop = nn.Dropout(p=self.drop_rate)
    dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(
        self.depths))]
    self.layers = nn.ModuleList()
    for i_layer in range(self.num_layers):
        layer = BasicLayer(dim=int(self.embed_dim * 2 ** i_layer),
            input_resolution=(patches_resolution[0] // 2 ** i_layer, 
            patches_resolution[1] // 2 ** i_layer), depth=self.depths[
            i_layer], num_heads=self.num_heads[i_layer], window_size=self.
            window_size, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale, drop=self.drop_rate, attn_drop=self.
            attn_drop_rate, drop_path=dpr[sum(self.depths[:i_layer]):sum(
            self.depths[:i_layer + 1])], norm_layer=self.norm_layer,
            downsample=PatchMerging if i_layer < self.num_layers - 1 else
            None, use_checkpoint=use_checkpoint, norm_before_mlp=self.
            norm_before_mlp)
        self.layers.append(layer)
    self.norm = self.norm_layer(self.num_features)
    self.avgpool = nn.AdaptiveAvgPool1d(1)
    self.maxpool = nn.AdaptiveMaxPool1d(1)
    SF = self.spec_size // 2 ** (len(self.depths) - 1) // self.patch_stride[0
        ] // self.freq_ratio
    self.tscam_conv = nn.Conv2d(in_channels=self.num_features, out_channels
        =self.num_classes, kernel_size=(SF, 3), padding=(0, 1))
    self.head = nn.Linear(num_classes, num_classes)
    if self.enable_fusion and self.fusion_type in ['daf_1d', 'aff_1d',
        'iaff_1d']:
        self.mel_conv1d = nn.Sequential(nn.Conv1d(64, 64, kernel_size=5,
            stride=3, padding=2), nn.BatchNorm1d(64))
        if self.fusion_type == 'daf_1d':
            self.fusion_model = DAF()
        elif self.fusion_type == 'aff_1d':
            self.fusion_model = AFF(channels=64, type='1D')
        elif self.fusion_type == 'iaff_1d':
            self.fusion_model = iAFF(channels=64, type='1D')
    self.apply(self._init_weights)
