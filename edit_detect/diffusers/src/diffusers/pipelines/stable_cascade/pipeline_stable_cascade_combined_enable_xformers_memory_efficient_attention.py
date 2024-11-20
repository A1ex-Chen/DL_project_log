def enable_xformers_memory_efficient_attention(self, attention_op: Optional
    [Callable]=None):
    self.decoder_pipe.enable_xformers_memory_efficient_attention(attention_op)
