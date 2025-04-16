def reshuffle_batches(self, indices, rng):
    """
        Permutes global batches
        :param indices: torch.tensor with batch indices
        :param rng: instance of torch.Generator
        """
    indices = indices.view(-1, self.global_batch_size)
    num_batches = indices.shape[0]
    order = torch.randperm(num_batches, generator=rng)
    indices = indices[order, :]
    indices = indices.view(-1)
    return indices
