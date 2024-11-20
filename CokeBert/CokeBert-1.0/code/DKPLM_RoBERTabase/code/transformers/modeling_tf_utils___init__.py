def __init__(self, config, initializer_range=0.02, **kwargs):
    super(TFSequenceSummary, self).__init__(**kwargs)
    self.summary_type = config.summary_type if hasattr(config,
        'summary_use_proj') else 'last'
    if self.summary_type == 'attn':
        raise NotImplementedError
    self.has_summary = hasattr(config, 'summary_use_proj'
        ) and config.summary_use_proj
    if self.has_summary:
        if hasattr(config, 'summary_proj_to_labels'
            ) and config.summary_proj_to_labels and config.num_labels > 0:
            num_classes = config.num_labels
        else:
            num_classes = config.hidden_size
        self.summary = tf.keras.layers.Dense(num_classes,
            kernel_initializer=get_initializer(initializer_range), name=
            'summary')
    self.has_activation = hasattr(config, 'summary_activation'
        ) and config.summary_activation == 'tanh'
    if self.has_activation:
        self.activation = tf.keras.activations.tanh
    self.has_first_dropout = hasattr(config, 'summary_first_dropout'
        ) and config.summary_first_dropout > 0
    if self.has_first_dropout:
        self.first_dropout = tf.keras.layers.Dropout(config.
            summary_first_dropout)
    self.has_last_dropout = hasattr(config, 'summary_last_dropout'
        ) and config.summary_last_dropout > 0
    if self.has_last_dropout:
        self.last_dropout = tf.keras.layers.Dropout(config.summary_last_dropout
            )
