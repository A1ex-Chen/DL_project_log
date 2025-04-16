def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):
    super().__init__()
    if isinstance(cfg, dict):
        self.yaml = cfg
    else:
        import yaml
        self.yaml_file = Path(cfg).name
        with open(cfg, encoding='ascii', errors='ignore') as f:
            self.yaml = yaml.safe_load(f)
    ch = self.yaml['ch'] = self.yaml.get('ch', ch)
    if nc and nc != self.yaml['nc']:
        LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
        self.yaml['nc'] = nc
    if anchors:
        LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
        self.yaml['anchors'] = round(anchors)
    self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])
    self.names = [str(i) for i in range(self.yaml['nc'])]
    self.inplace = self.yaml.get('inplace', True)
    m = self.model[-1]
    if isinstance(m, Detect):
        s = 256
        m.inplace = self.inplace
        m.stride = torch.tensor([(s / x.shape[-2]) for x in self.forward(
            torch.zeros(1, ch, s, s))])
        check_anchor_order(m)
        m.anchors /= m.stride.view(-1, 1, 1)
        self.stride = m.stride
        self._initialize_biases()
    initialize_weights(self)
    self.info()
    LOGGER.info('')
