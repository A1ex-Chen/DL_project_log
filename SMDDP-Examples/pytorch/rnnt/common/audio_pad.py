def pad(self, pad_size, symmetric=False):
    """Add zero padding to the sample.

        The pad size is given in number of samples. If symmetric=True,
        `pad_size` will be added to both sides. If false, `pad_size` zeros
        will be added only to the end.
        """
    self._samples = np.pad(self._samples, (pad_size if symmetric else 0,
        pad_size), mode='constant')
