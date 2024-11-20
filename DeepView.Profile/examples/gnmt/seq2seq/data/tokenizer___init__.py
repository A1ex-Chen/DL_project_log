def __init__(self, vocab_fname=None, pad=1, separator='@@'):
    """
        Constructor for the Tokenizer class.

        :param vocab_fname: path to the file with vocabulary
        :param pad: pads vocabulary to a multiple of 'pad' tokens
        :param separator: tokenization separator
        """
    if vocab_fname:
        self.separator = separator
        logging.info(f'Building vocabulary from {vocab_fname}')
        vocab = [config.PAD_TOKEN, config.UNK_TOKEN, config.BOS_TOKEN,
            config.EOS_TOKEN]
        with open(vocab_fname) as vfile:
            for line in vfile:
                vocab.append(line.strip())
        self.pad_vocabulary(vocab, pad)
        self.vocab_size = len(vocab)
        logging.info(f'Size of vocabulary: {self.vocab_size}')
        self.tok2idx = defaultdict(partial(int, config.UNK))
        for idx, token in enumerate(vocab):
            self.tok2idx[token] = idx
        self.idx2tok = {}
        for key, value in self.tok2idx.items():
            self.idx2tok[value] = key
