def __init__(self, cfg_path, DATA_DIR):
    self.config = {}
    registry.register('configuration', self)
    config = OmegaConf.load(cfg_path)
    model_config = self.build_model_config(config, DATA_DIR=DATA_DIR)
    preprocess_config = self.build_preprocess_config(config)
    self.config = OmegaConf.merge(model_config, preprocess_config)
