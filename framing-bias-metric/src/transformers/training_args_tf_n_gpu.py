@property
@tf_required
def n_gpu(self) ->int:
    """
        The number of replicas (CPUs, GPUs or TPU cores) used in this training.
        """
    warnings.warn(
        'The n_gpu argument is deprecated and will be removed in a future version, use n_replicas instead.'
        , FutureWarning)
    return self._setup_strategy.num_replicas_in_sync
