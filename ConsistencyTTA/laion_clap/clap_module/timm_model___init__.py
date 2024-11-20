def __init__(self, model_name, embed_dim, image_size=224, pool='avg', proj=
    'linear', drop=0.0, pretrained=False):
    super().__init__()
    if timm is None:
        raise RuntimeError('Please `pip install timm` to use timm models.')
    self.image_size = to_2tuple(image_size)
    self.trunk = timm.create_model(model_name, pretrained=pretrained)
    feat_size = self.trunk.default_cfg.get('pool_size', None)
    feature_ndim = 1 if not feat_size else 2
    if pool in ('abs_attn', 'rot_attn'):
        assert feature_ndim == 2
        self.trunk.reset_classifier(0, global_pool='')
    else:
        reset_kwargs = dict(global_pool=pool) if pool else {}
        self.trunk.reset_classifier(0, **reset_kwargs)
    prev_chs = self.trunk.num_features
    head_layers = OrderedDict()
    if pool == 'abs_attn':
        head_layers['pool'] = AbsAttentionPool2d(prev_chs, feat_size=
            feat_size, out_features=embed_dim)
        prev_chs = embed_dim
    elif pool == 'rot_attn':
        head_layers['pool'] = RotAttentionPool2d(prev_chs, out_features=
            embed_dim)
        prev_chs = embed_dim
    else:
        assert proj, 'projection layer needed if non-attention pooling is used.'
    if proj == 'linear':
        head_layers['drop'] = nn.Dropout(drop)
        head_layers['proj'] = nn.Linear(prev_chs, embed_dim)
    elif proj == 'mlp':
        head_layers['mlp'] = Mlp(prev_chs, 2 * embed_dim, embed_dim, drop=drop)
    self.head = nn.Sequential(head_layers)
