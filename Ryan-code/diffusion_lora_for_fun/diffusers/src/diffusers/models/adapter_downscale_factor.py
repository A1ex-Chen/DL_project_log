@property
def downscale_factor(self):
    """The downscale factor applied in the T2I-Adapter's initial pixel unshuffle operation. If an input image's dimensions are
        not evenly divisible by the downscale_factor then an exception will be raised.
        """
    return self.adapter.unshuffle.downscale_factor
