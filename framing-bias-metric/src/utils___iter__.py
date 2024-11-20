def __iter__(self) ->Iterable:
    g = torch.Generator()
    g.manual_seed(self.epoch)
    sortish_data = [self.dataset.src_lens[i] for i in self.available_indices]
    sortish_indices = sortish_sampler_indices(sortish_data, self.batch_size,
        shuffle=self.shuffle)
    indices = [self.available_indices[i] for i in sortish_indices]
    assert len(indices) == self.num_samples
    return iter(indices)
