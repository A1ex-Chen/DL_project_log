def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, model=None, imgsz=(
    640, 640)):
    super().__init__()
    if isinstance(cfg, dict):
        self.yaml = cfg
    else:
        import yaml
        self.yaml_file = Path(cfg).name
        with open(cfg) as f:
            self.yaml = yaml.load(f, Loader=yaml.FullLoader)
    if nc and nc != self.yaml['nc']:
        LOGGER.info(f"Overriding {cfg} nc={self.yaml['nc']} with nc={nc}")
        self.yaml['nc'] = nc
    self.model, self.savelist = parse_model(deepcopy(self.yaml), ch=[ch],
        model=model, imgsz=imgsz)
