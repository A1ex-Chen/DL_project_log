@classmethod
def from_name(cls, model_name: Text, model_weights_path: Text=None,
    weights_format: Text='saved_model', overrides: Dict[Text, Any]=None):
    """Construct an EfficientNet model from a predefined model name.

    E.g., `EfficientNet.from_name('efficientnet-b0')`.

    Args:
      model_name: the predefined model name
      model_weights_path: the path to the weights (h5 file or saved model dir)
      weights_format: the model weights format. One of 'saved_model', 'h5',
       or 'checkpoint'.
      overrides: (optional) a dict containing keys that can override config

    Returns:
      A constructed EfficientNet instance.
    """
    model_configs = dict(MODEL_CONFIGS)
    overrides = dict(overrides) if overrides else {}
    model_configs.update(overrides.pop('model_config', {}))
    if model_name not in model_configs:
        raise ValueError('Unknown model name {}'.format(model_name))
    config = model_configs[model_name]
    model = cls(config=config, overrides=overrides)
    if model_weights_path:
        load_weights(model, model_weights_path, weights_format=weights_format)
    return model
