def __init__(self, size: int):
    """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
        """
    self._size = size
    assert size > 0
    self._rank = comm.get_rank()
    self._world_size = comm.get_world_size()
    self._local_indices = self._get_local_indices(size, self._world_size,
        self._rank)
