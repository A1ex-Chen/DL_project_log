@property
@tf_required
def n_replicas(self) ->int:
    """
        The number of replicas (CPUs, GPUs or TPU cores) used in this training.
        """
    return self._setup_strategy.num_replicas_in_sync
