def _infinite_indices(self):
    g = torch.Generator()
    g.manual_seed(self._seed)
    while True:
        indices = self._get_epoch_indices(g)
        if self._shuffle:
            randperm = torch.randperm(len(indices), generator=g)
            yield from indices[randperm].tolist()
        else:
            yield from indices.tolist()
