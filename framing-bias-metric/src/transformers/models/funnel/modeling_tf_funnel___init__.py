def __init__(self, config, *inputs, **kwargs):
    super().__init__(config, *inputs, **kwargs)
    self.num_labels = config.num_labels
    self.funnel = TFFunnelMainLayer(config, name='funnel')
    self.qa_outputs = tf.keras.layers.Dense(config.num_labels,
        kernel_initializer=get_initializer(config.initializer_range), name=
        'qa_outputs')
