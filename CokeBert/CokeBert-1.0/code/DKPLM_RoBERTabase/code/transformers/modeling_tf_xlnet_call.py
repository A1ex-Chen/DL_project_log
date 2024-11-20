def call(self, inputs, **kwargs):
    transformer_outputs = self.transformer(inputs, **kwargs)
    sequence_output = transformer_outputs[0]
    logits = self.qa_outputs(sequence_output)
    start_logits, end_logits = tf.split(logits, 2, axis=-1)
    start_logits = tf.squeeze(start_logits, axis=-1)
    end_logits = tf.squeeze(end_logits, axis=-1)
    outputs = (start_logits, end_logits) + transformer_outputs[1:]
    return outputs
