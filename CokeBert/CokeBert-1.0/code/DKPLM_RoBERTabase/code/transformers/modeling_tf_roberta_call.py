def call(self, inputs, **kwargs):
    outputs = self.roberta(inputs, **kwargs)
    sequence_output = outputs[0]
    sequence_output = self.dropout(sequence_output, training=kwargs.get(
        'training', False))
    logits = self.classifier(sequence_output)
    outputs = (logits,) + outputs[2:]
    return outputs
