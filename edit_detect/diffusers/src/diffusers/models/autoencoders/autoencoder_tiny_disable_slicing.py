def disable_slicing(self) ->None:
    """
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
    self.use_slicing = False
