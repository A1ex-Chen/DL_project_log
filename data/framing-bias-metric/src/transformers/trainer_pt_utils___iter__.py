def __iter__(self):
    indices = list(range(len(self.dataset)))
    indices += indices[:self.total_size - len(indices)]
    assert len(indices
        ) == self.total_size, f'Indices length {len(indices)} and total size {self.total_size} mismatched'
    indices = indices[self.rank * self.num_samples:(self.rank + 1) * self.
        num_samples]
    assert len(indices
        ) == self.num_samples, f'Indices length {len(indices)} and sample number {self.num_samples} mismatched'
    return iter(indices)
