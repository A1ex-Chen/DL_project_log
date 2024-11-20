def __init__(self, config, *inputs, **kwargs):
    super(TFRobertaForTokenClassification, self).__init__(config, *inputs,
        **kwargs)
    self.num_labels = config.num_labels
    self.roberta = TFRobertaMainLayer(config, name='roberta')
    self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
    self.classifier = tf.keras.layers.Dense(config.num_labels,
        kernel_initializer=get_initializer(config.initializer_range), name=
        'classifier')
