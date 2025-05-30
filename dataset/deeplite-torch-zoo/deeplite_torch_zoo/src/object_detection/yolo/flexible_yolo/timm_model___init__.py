def __init__(self, backbone_name, nc=80, anchors=None, neck_cfg=None,
    custom_head=None):
    nn.Module.__init__(self)
    self.yaml = None
    head_config = {'nc': nc, 'anchors': anchors if anchors is not None else
        ANCHOR_REGISTRY.get('default')()}
    default_neck_cfg = {'v5FPN': {'channel_outs': [512, 256, 256],
        'version': 's'}, 'v5PAN': {'channel_outs': [256, 512, 512, 1024],
        'version': 's'}}
    if neck_cfg is None:
        neck_cfg = default_neck_cfg
    head_cls = Detect
    if custom_head is not None:
        if custom_head not in HEAD_NAME_MAP:
            raise ValueError(
                f'Incorrect YOLO head name {custom_head}. Choices: {list(HEAD_NAME_MAP.keys())}'
                )
        head_cls = HEAD_NAME_MAP[custom_head]
    self.backbone = TimmWrapperBackbone(backbone_name)
    ch_in = self.backbone.out_shape
    self.necks = nn.ModuleList()
    for neck_name, neck_params in neck_cfg.items():
        neck_params['ch'] = ch_in
        neck = build_neck(neck_name, **neck_params)
        ch_in = neck.out_shape
        self.necks.append(neck)
    head_config['ch'] = ch_in
    if head_cls != Detect:
        head_config.pop('anchors')
    self.detection = head_cls(**head_config)
    self._init_head()
    initialize_weights(self)
    self._is_fused = False
