def __init__(self, config: Dict[Text, Any]=None, overrides: Dict[Text, Any]
    =None):
    """Create an EfficientNet model.

    Args:
      config: (optional) the main model parameters to create the model
      overrides: (optional) a dict containing keys that can override
                 config
    """
    overrides = overrides or {}
    is_training = overrides.pop('is_training', False)
    config = config or build_dict(name='ModelConfig')
    self.config = config
    self.config.update(overrides)
    input_channels = self.config['input_channels']
    model_name = self.config['model_name']
    input_shape = None, None, input_channels
    image_input = tf.keras.layers.Input(shape=input_shape)
    if is_training:
        beta_input = tf.keras.layers.Input(shape=(1, 1, 1))
        inputs = image_input, beta_input
        output = efficientnet(inputs, self.config)
    else:
        inputs = [image_input]
        output = efficientnet(inputs, self.config)
    output = tf.cast(output, tf.float32)
    super(EfficientNet, self).__init__(inputs=inputs, outputs=output, name=
        model_name)
