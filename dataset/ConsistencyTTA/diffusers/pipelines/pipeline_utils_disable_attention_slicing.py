def disable_attention_slicing(self):
    """
        Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
        back to computing attention in one step.
        """
    self.enable_attention_slicing(None)
