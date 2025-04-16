def enable_tiling(self, use_tiling: bool=True):
    """
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful to save a large amount of memory and to allow
        the processing of larger images.
        """
    self.use_tiling = use_tiling
