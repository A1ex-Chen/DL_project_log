@property
def global_batch_size(self) ->int:
    """The batch size, multiplied by the number of replicas (if configured)."""
    return self._batch_size * self._num_gpus
