def ff_chunk(self, attention_output):
    ffn_output = self.ffn(attention_output)
    ffn_output = self.activation(ffn_output)
    ffn_output = self.ffn_output(ffn_output)
    return ffn_output
