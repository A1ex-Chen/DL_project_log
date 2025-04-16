def call(self, inputs, **kwargs):
    distilbert_output = self.distilbert(inputs, **kwargs)
    hidden_states = distilbert_output[0]
    hidden_states = self.dropout(hidden_states, training=kwargs.get(
        'training', False))
    logits = self.qa_outputs(hidden_states)
    start_logits, end_logits = tf.split(logits, 2, axis=-1)
    start_logits = tf.squeeze(start_logits, axis=-1)
    end_logits = tf.squeeze(end_logits, axis=-1)
    outputs = (start_logits, end_logits) + distilbert_output[1:]
    return outputs
