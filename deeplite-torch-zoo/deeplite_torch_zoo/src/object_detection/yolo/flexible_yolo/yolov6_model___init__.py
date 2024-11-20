def __init__(self, model_config, nc=80, anchors=None, custom_head=None,
    width_mul=None, depth_mul=None, is_lite=False):
    """
        :param model_config:
        """
    nn.Module.__init__(self)
    self.yaml = None
    self._is_lite = is_lite
    head_config = {'nc': nc, 'anchors': anchors if anchors is not None else
        ANCHOR_REGISTRY.get('default')() if not self._is_lite else
        ANCHOR_REGISTRY.get('default_p6')()}
    head_cls = Detect
    if custom_head is not None:
        if custom_head not in HEAD_NAME_MAP:
            raise ValueError(
                f'Incorrect YOLO head name {custom_head}. Choices: {list(HEAD_NAME_MAP.keys())}'
                )
        head_cls = HEAD_NAME_MAP[custom_head]
    cfg = Config.fromfile(model_config)
    if not hasattr(cfg, 'training_mode'):
        setattr(cfg, 'training_mode', 'repvgg')
    if width_mul is not None:
        cfg.model.width_multiple = width_mul
    if depth_mul is not None:
        cfg.model.depth_multiple = depth_mul
    builder_fn = build_network if not self._is_lite else build_network_lite
    self.backbone, self.neck, channel_counts = builder_fn(cfg)
    self.necks = nn.ModuleList()
    self.necks.append(self.neck)
    head_config['ch'] = channel_counts
    if head_cls != Detect:
        head_config.pop('anchors')
    self.detection = head_cls(**head_config)
    self._init_head()
    initialize_weights(self)
    self._is_fused = False
