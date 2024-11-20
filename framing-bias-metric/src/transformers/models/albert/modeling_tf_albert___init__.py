def __init__(self, config, *inputs, **kwargs):
    super().__init__(config, *inputs, **kwargs)
    self.albert = TFAlbertMainLayer(config, name='albert')
    self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
    self.classifier = tf.keras.layers.Dense(1, kernel_initializer=
        get_initializer(config.initializer_range), name='classifier')
