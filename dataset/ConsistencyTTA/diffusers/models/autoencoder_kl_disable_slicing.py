def disable_slicing(self):
    """
        Disable sliced VAE decoding. If `enable_slicing` was previously invoked, this method will go back to computing
        decoding in one step.
        """
    self.use_slicing = False
