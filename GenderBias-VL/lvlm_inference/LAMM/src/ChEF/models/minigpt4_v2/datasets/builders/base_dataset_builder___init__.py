def __init__(self, cfg=None):
    super().__init__()
    if cfg is None:
        self.config = load_dataset_config(self.default_config_path())
    elif isinstance(cfg, str):
        self.config = load_dataset_config(cfg)
    else:
        self.config = cfg
    self.data_type = self.config.data_type
    self.vis_processors = {'train': BaseProcessor(), 'eval': BaseProcessor()}
    self.text_processors = {'train': BaseProcessor(), 'eval': BaseProcessor()}
