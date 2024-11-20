def __init__(self, in_chans=3, num_classes=1000, embed_dim=96, depths=(1, 2,
    8, 2), mlp_ratio=2.0, n_div=4, patch_size=4, patch_stride=4,
    patch_size2=2, patch_stride2=2, patch_norm=True, feature_dim=1280,
    drop_path_rate=0.1, layer_scale_init_value=0, norm_layer='bn', act=
    'relu', features_only=False, init_cfg=None, pconv_fw_type='split_cat'):
    super().__init__()
    if norm_layer == 'bn':
        norm_layer = nn.BatchNorm2d
    else:
        raise NotImplementedError
    act_layer = get_activation(act)
    if not features_only:
        self.num_classes = num_classes
    self.num_stages = len(depths)
    self.embed_dim = embed_dim
    self.patch_norm = patch_norm
    self.num_features = int(embed_dim * 2 ** (self.num_stages - 1))
    self.mlp_ratio = mlp_ratio
    self.depths = depths
    self.patch_embed = PatchEmbed(patch_size=patch_size, patch_stride=
        patch_stride, in_chans=in_chans, embed_dim=embed_dim, norm_layer=
        norm_layer if self.patch_norm else None)
    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
    stages_list = []
    for i_stage in range(self.num_stages):
        stage = BasicStage(dim=int(embed_dim * 2 ** i_stage), n_div=n_div,
            depth=depths[i_stage], mlp_ratio=self.mlp_ratio, drop_path=dpr[
            sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
            layer_scale_init_value=layer_scale_init_value, norm_layer=
            norm_layer, act_layer=act_layer, pconv_fw_type=pconv_fw_type)
        stages_list.append(stage)
        if i_stage < self.num_stages - 1:
            stages_list.append(PatchMerging(patch_size2=patch_size2,
                patch_stride2=patch_stride2, dim=int(embed_dim * 2 **
                i_stage), norm_layer=norm_layer))
    self.stages = nn.Sequential(*stages_list)
    if features_only:
        self.forward = self.forward_features
        self.out_indices = [0, 2, 4, 6]
        for i_emb, i_layer in enumerate(self.out_indices):
            if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                raise NotImplementedError
            else:
                layer = norm_layer(int(embed_dim * 2 ** i_emb))
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
    else:
        self.avgpool_pre_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.
            Conv2d(self.num_features, feature_dim, 1, bias=False), act_layer)
        self.head = nn.Linear(feature_dim, num_classes
            ) if num_classes > 0 else nn.Identity()
    self.apply(self._init_weights)
    self.init_cfg = copy.deepcopy(init_cfg)
