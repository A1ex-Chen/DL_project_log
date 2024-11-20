@abstractmethod
def subsample(self, indicator, batch_size, **params):
    """Returns subsample of entries in indicator.

    Args:
      indicator: boolean tensor of shape [N] whose True entries can be sampled.
      batch_size: desired batch size.
      **params: additional keyword arguments for specific implementations of
          the MinibatchSampler.

    Returns:
      sample_indicator: boolean tensor of shape [N] whose True entries have been
      sampled. If sum(indicator) >= batch_size, sum(is_sampled) = batch_size
    """
    pass
