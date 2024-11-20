def __len__(self):
    """Returns the length of the batch sampler's sampler."""
    return len(self.batch_sampler.sampler)
