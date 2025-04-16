def disable_tiling(self) ->None:
    """
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
    self.enable_tiling(False)
