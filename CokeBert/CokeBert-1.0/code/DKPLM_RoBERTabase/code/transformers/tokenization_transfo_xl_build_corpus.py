def build_corpus(self, path, dataset):
    self.dataset = dataset
    if self.dataset in ['ptb', 'wt2', 'enwik8', 'text8']:
        self.vocab.count_file(os.path.join(path, 'train.txt'))
        self.vocab.count_file(os.path.join(path, 'valid.txt'))
        self.vocab.count_file(os.path.join(path, 'test.txt'))
    elif self.dataset == 'wt103':
        self.vocab.count_file(os.path.join(path, 'train.txt'))
    elif self.dataset == 'lm1b':
        train_path_pattern = os.path.join(path,
            '1-billion-word-language-modeling-benchmark-r13output',
            'training-monolingual.tokenized.shuffled', 'news.en-*')
        train_paths = glob.glob(train_path_pattern)
    self.vocab.build_vocab()
    if self.dataset in ['ptb', 'wt2', 'wt103']:
        self.train = self.vocab.encode_file(os.path.join(path, 'train.txt'),
            ordered=True)
        self.valid = self.vocab.encode_file(os.path.join(path, 'valid.txt'),
            ordered=True)
        self.test = self.vocab.encode_file(os.path.join(path, 'test.txt'),
            ordered=True)
    elif self.dataset in ['enwik8', 'text8']:
        self.train = self.vocab.encode_file(os.path.join(path, 'train.txt'),
            ordered=True, add_eos=False)
        self.valid = self.vocab.encode_file(os.path.join(path, 'valid.txt'),
            ordered=True, add_eos=False)
        self.test = self.vocab.encode_file(os.path.join(path, 'test.txt'),
            ordered=True, add_eos=False)
    elif self.dataset == 'lm1b':
        self.train = train_paths
        self.valid = self.vocab.encode_file(os.path.join(path, 'valid.txt'),
            ordered=False, add_double_eos=True)
        self.test = self.vocab.encode_file(os.path.join(path, 'test.txt'),
            ordered=False, add_double_eos=True)
