def __init__(self, dataset, batch_size, shuffle, validation_split,
    num_workers, collate_fn=default_collate):
    self.validation_split = validation_split
    self.shuffle = shuffle
    self.batch_idx = 0
    self.n_samples = len(dataset)
    self.sampler, self.valid_sampler = self._split_sampler(self.
        validation_split)
    self.init_kwargs = {'dataset': dataset, 'batch_size': batch_size,
        'shuffle': self.shuffle, 'collate_fn': collate_fn, 'num_workers':
        num_workers}
    super().__init__(sampler=self.sampler, **self.init_kwargs)
