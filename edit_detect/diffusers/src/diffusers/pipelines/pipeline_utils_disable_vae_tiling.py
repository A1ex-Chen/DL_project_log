def disable_vae_tiling(self):
    """
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
    self.vae.disable_tiling()
