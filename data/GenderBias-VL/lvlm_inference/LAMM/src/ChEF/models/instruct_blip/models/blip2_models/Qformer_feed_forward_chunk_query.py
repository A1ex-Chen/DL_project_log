def feed_forward_chunk_query(self, attention_output):
    intermediate_output = self.intermediate_query(attention_output)
    layer_output = self.output_query(intermediate_output, attention_output)
    return layer_output
