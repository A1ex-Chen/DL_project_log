def call(self, inputs, **kwargs):
    transformer_outputs = self.transformer(inputs, **kwargs)
    hidden_states = transformer_outputs[0]
    lm_logits = self.lm_head(hidden_states)
    outputs = (lm_logits,) + transformer_outputs[1:]
    return outputs
