def get_iterator(self, split, *args, **kwargs):
    if split == 'train':
        if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8']:
            data_iter = LMOrderedIterator(self.train, *args, **kwargs)
        elif self.dataset == 'lm1b':
            kwargs['shuffle'] = True
            data_iter = LMMultiFileIterator(self.train, self.vocab, *args,
                **kwargs)
    elif split in ['valid', 'test']:
        data = self.valid if split == 'valid' else self.test
        if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8']:
            data_iter = LMOrderedIterator(data, *args, **kwargs)
        elif self.dataset == 'lm1b':
            data_iter = LMShuffledIterator(data, *args, **kwargs)
    else:
        data_iter = None
        raise ValueError(f'Split not recognized: {split}')
    return data_iter
