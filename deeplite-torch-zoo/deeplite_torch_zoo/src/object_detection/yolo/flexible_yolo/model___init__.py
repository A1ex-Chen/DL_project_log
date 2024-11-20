def __init__(self, model_config, nc=None, backbone_kwargs=None, neck_kwargs
    =None, custom_head=None):
    """
        :param model_config:
        """
    nn.Module.__init__(self)
    self.yaml = None
    head_cls = Detect
    if custom_head is not None:
        if custom_head not in HEAD_NAME_MAP:
            raise ValueError(
                f'Incorrect YOLO head name {custom_head}. Choices: {list(HEAD_NAME_MAP.keys())}'
                )
        head_cls = HEAD_NAME_MAP[custom_head]
    if type(model_config) is str:
        model_config = yaml.load(open(model_config, 'r'), Loader=yaml.
            SafeLoader)
    model_config = Dict(model_config)
    if nc is not None:
        model_config.head.nc = nc
    if backbone_kwargs is not None:
        model_config.backbone.update(Dict(backbone_kwargs))
    backbone_type = model_config.backbone.pop('type')
    self.backbone = build_backbone(backbone_type, **model_config.backbone)
    ch_in = self.backbone.out_shape
    self.necks = nn.ModuleList()
    necks_config = model_config.neck
    if neck_kwargs is not None:
        necks_config.update(Dict(neck_kwargs))
    for neck_name, neck_params in necks_config.items():
        neck_params['ch'] = ch_in
        neck = build_neck(neck_name, **neck_params)
        ch_in = neck.out_shape
        self.necks.append(neck)
    model_config.head['ch'] = ch_in
    if head_cls != Detect:
        model_config.head.pop('anchors')
    self.detection = head_cls(**model_config.head)
    self._init_head()
    initialize_weights(self)
    self._is_fused = False
