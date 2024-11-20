def __init__(self, cfg='yolor-csp-c.yaml', ch=3, nc=None, anchors=None):
    super(Model, self).__init__()
    self.traced = False
    if isinstance(cfg, dict):
        self.yaml = cfg
    else:
        import yaml
        self.yaml_file = Path(cfg).name
        with open(cfg) as f:
            self.yaml = yaml.load(f, Loader=yaml.SafeLoader)
    ch = self.yaml['ch'] = self.yaml.get('ch', ch)
    if nc and nc != self.yaml['nc']:
        logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
        self.yaml['nc'] = nc
    if anchors:
        logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
        self.yaml['anchors'] = round(anchors)
    self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])
    self.names = [str(i) for i in range(self.yaml['nc'])]
    m = self.model[-1]
    if isinstance(m, Detect):
        s = 256
        m.stride = torch.tensor([(s / x.shape[-2]) for x in self.forward(
            torch.zeros(1, ch, s, s))])
        check_anchor_order(m)
        m.anchors /= m.stride.view(-1, 1, 1)
        self.stride = m.stride
        self._initialize_biases()
    if isinstance(m, IDetect):
        s = 256
        m.stride = torch.tensor([(s / x.shape[-2]) for x in self.forward(
            torch.zeros(1, ch, s, s))])
        check_anchor_order(m)
        m.anchors /= m.stride.view(-1, 1, 1)
        self.stride = m.stride
        self._initialize_biases()
    if isinstance(m, IAuxDetect):
        s = 256
        m.stride = torch.tensor([(s / x.shape[-2]) for x in self.forward(
            torch.zeros(1, ch, s, s))[:4]])
        check_anchor_order(m)
        m.anchors /= m.stride.view(-1, 1, 1)
        self.stride = m.stride
        self._initialize_aux_biases()
    if isinstance(m, IBin):
        s = 256
        m.stride = torch.tensor([(s / x.shape[-2]) for x in self.forward(
            torch.zeros(1, ch, s, s))])
        check_anchor_order(m)
        m.anchors /= m.stride.view(-1, 1, 1)
        self.stride = m.stride
        self._initialize_biases_bin()
    if isinstance(m, IKeypoint):
        s = 256
        m.stride = torch.tensor([(s / x.shape[-2]) for x in self.forward(
            torch.zeros(1, ch, s, s))])
        check_anchor_order(m)
        m.anchors /= m.stride.view(-1, 1, 1)
        self.stride = m.stride
        self._initialize_biases_kpt()
    initialize_weights(self)
    self.info()
    logger.info('')
