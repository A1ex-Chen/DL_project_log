def __init__(self, config, *inputs, **kwargs):
    super(TFDistilBertForQuestionAnswering, self).__init__(config, *inputs,
        **kwargs)
    self.distilbert = TFDistilBertMainLayer(config, name='distilbert')
    self.qa_outputs = tf.keras.layers.Dense(config.num_labels,
        kernel_initializer=get_initializer(config.initializer_range), name=
        'qa_outputs')
    assert config.num_labels == 2
    self.dropout = tf.keras.layers.Dropout(config.qa_dropout)
