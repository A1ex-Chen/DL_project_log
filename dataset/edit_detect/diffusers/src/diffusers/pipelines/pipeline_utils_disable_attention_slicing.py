def disable_attention_slicing(self):
    """
        Disable sliced attention computation. If `enable_attention_slicing` was previously called, attention is
        computed in one step.
        """
    self.enable_attention_slicing(None)
