def ff_chunk(self, attn_output):
    intermediate_output = self.intermediate(attn_output)
    layer_output = self.output(intermediate_output, attn_output)
    return layer_output
