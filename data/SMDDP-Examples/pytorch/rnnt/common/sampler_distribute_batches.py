def distribute_batches(self, indices):
    """
        Assigns batches to workers.
        Consecutive ranks are getting consecutive batches.
        :param indices: torch.tensor with batch indices
        """
    assert len(indices) == self.num_samples
    indices = indices.view(-1, self.batch_size)
    indices = indices[self.rank::self.world_size].contiguous()
    indices = indices.view(-1)
    indices = indices.tolist()
    assert len(indices) == self.num_samples // self.world_size
    return indices
