def feed_forward_chunk(self, attention_output):
    intermediate_output = self.intermediate(attention_output)
    layer_output = self.output(intermediate_output, attention_output)
    return layer_output
