def disable_xformers_memory_efficient_attention(self):
    """
        Disable memory efficient attention as implemented in xformers.
        """
    self.set_use_memory_efficient_attention_xformers(False)
