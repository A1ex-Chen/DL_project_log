def _from_yaml(self, cfg, ch, nc, verbose):
    """Set YOLOv8 model configurations and define the model architecture."""
    self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
    ch = self.yaml['ch'] = self.yaml.get('ch', ch)
    if nc and nc != self.yaml['nc']:
        LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
        self.yaml['nc'] = nc
    elif not nc and not self.yaml.get('nc', None):
        raise ValueError(
            'nc not specified. Must specify nc in model.yaml or function arguments.'
            )
    self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose
        =verbose)
    self.stride = torch.Tensor([1])
    self.names = {i: f'{i}' for i in range(self.yaml['nc'])}
    self.info()
