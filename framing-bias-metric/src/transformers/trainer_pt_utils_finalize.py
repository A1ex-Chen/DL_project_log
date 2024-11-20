def finalize(self):
    """
        Return the properly gathered arrays and truncate to the number of samples (since the sampler added some extras
        to get each process a dataset of the same length).
        """
    if self._storage is None:
        return
    if self._offsets[0] != self.process_length:
        logger.warn(
            'Not all data has been set. Are you sure you passed all values?')
    return nested_truncate(self._storage, self.num_samples)
