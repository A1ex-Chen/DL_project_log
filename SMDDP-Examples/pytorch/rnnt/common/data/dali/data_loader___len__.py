def __len__(self):
    """
        Number of batches handled by each GPU.
        """
    if self.drop_last:
        assert self._shard_size(
            ) % self.batch_size == 0, f'{self._shard_size()} {self.batch_size}'
    return int(math.ceil(self._shard_size() / self.batch_size))
