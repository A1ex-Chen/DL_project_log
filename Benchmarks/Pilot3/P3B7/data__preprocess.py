def _preprocess(self, raw_data):
    print('Preprocessing data...')
    self._make_processed_dirs()
    with open(raw_data, 'rb') as f:
        x_train = np.flip(pickle.load(f), 1)
        y_train = pickle.load(f)
        x_valid = np.flip(pickle.load(f), 1)
        y_valid = pickle.load(f)
    corpus = Tokenizer(x_train, x_valid)
    self.num_vocab = len(corpus.vocab)
    self._save_split('train', corpus.train, y_train)
    self._save_split('valid', corpus.valid, y_valid)
    self._save_vocab(corpus.vocab)
    print('Done!')
