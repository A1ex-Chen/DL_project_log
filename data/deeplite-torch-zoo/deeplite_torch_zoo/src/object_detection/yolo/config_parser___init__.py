def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None,
    activation_type=None, depth_mul=None, width_mul=None, channel_divisor=8,
    max_channels=None, custom_head=None, verbose=False):
    super().__init__()
    if isinstance(cfg, dict):
        self.yaml = cfg
    else:
        import yaml
        self.yaml_file = Path(cfg).name
        with open(cfg, encoding='ascii', errors='ignore') as f:
            self.yaml = yaml.safe_load(f)
    if custom_head is not None:
        if custom_head not in HEAD_NAME_MAP:
            raise ValueError(
                f'Incorrect YOLO head name {custom_head}. Choices: {list(HEAD_NAME_MAP.keys())}'
                )
        self.yaml['head'][-1][2] = HEAD_NAME_MAP[custom_head].__name__
    self.nc = nc
    ch = self.yaml['ch'] = self.yaml.get('ch', ch)
    if nc and nc != self.yaml['nc']:
        LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
        self.yaml['nc'] = nc
    if anchors:
        LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
        self.yaml['anchors'] = round(anchors)
    self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch],
        activation_type=activation_type, depth_mul=depth_mul, width_mul=
        width_mul, yolo_channel_divisor=channel_divisor, max_channels=
        max_channels)
    self.names = [str(i) for i in range(self.yaml['nc'])]
    self.inplace = self.yaml.get('inplace', True)
    self._init_head(ch)
    initialize_weights(self)
    self._is_fused = False
    if verbose:
        self.info()
