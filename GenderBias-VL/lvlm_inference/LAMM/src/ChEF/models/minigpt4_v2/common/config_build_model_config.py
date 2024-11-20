@staticmethod
def build_model_config(config, **kwargs):
    model = config.get('model', None)
    assert model is not None, 'Missing model configuration file.'
    model_cls = registry.get_model_class(model.arch)
    assert model_cls is not None, f"Model '{model.arch}' has not been registered."
    model_type = kwargs.get('model.model_type', None)
    if not model_type:
        model_type = model.get('model_type', None)
    assert model_type is not None, 'Missing model_type.'
    model_config_path = model_cls.default_config_path(model_type=model_type)
    model_config = OmegaConf.create()
    model_config = OmegaConf.merge(model_config, OmegaConf.load(
        model_config_path), {'model': config['model']})
    return model_config
