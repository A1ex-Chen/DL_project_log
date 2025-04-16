def _split_sampler(self, split):
    if split == 0.0:
        return None, None
    idx_full = np.arange(self.n_samples)
    np.random.seed(0)
    np.random.shuffle(idx_full)
    if isinstance(split, int):
        assert split > 0
        assert split < self.n_samples, 'validation set size is configured to be larger than entire dataset.'
        len_valid = split
    else:
        len_valid = int(self.n_samples * split)
    valid_idx = idx_full[0:len_valid]
    train_idx = np.delete(idx_full, np.arange(0, len_valid))
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    self.shuffle = False
    self.n_samples = len(train_idx)
    return train_sampler, valid_sampler
