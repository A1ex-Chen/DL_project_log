@cached_property
def available_indices(self) ->np.array:
    indices = list(range(len(self.dataset)))
    indices += indices[:self.total_size - len(indices)]
    assert len(indices) == self.total_size
    available_indices = indices[self.rank:self.total_size:self.num_replicas]
    return available_indices
