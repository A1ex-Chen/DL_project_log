def disable_vae_slicing(self):
    """
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
    self.vae.disable_slicing()
