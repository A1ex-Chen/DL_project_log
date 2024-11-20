def set_default_attn_processor(self):
    """
        Disables custom attention processors and sets the default attention implementation.
        """
    self.set_attn_processor(AttnProcessor())
