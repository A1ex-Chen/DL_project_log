def disable_tiling(self):
    """
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously invoked, this method will go back to
        computing decoding in one step.
        """
    self.enable_tiling(False)
