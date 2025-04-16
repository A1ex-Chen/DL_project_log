def __init__(self, src_fname, tgt_fname, tokenizer, min_len, max_len, sort=
    False, max_size=None):
    """
        Constructor for the LazyParallelDataset.
        Tokenization is done on the fly.

        :param src_fname: path to the file with src language data
        :param tgt_fname: path to the file with tgt language data
        :param tokenizer: tokenizer
        :param min_len: minimum sequence length
        :param max_len: maximum sequence length
        :param sort: sorts dataset by sequence length
        :param max_size: loads at most 'max_size' samples from the input file,
            if None loads the entire dataset
        """
    self.min_len = min_len
    self.max_len = max_len
    self.parallel = True
    self.sorted = False
    self.tokenizer = tokenizer
    self.raw_src = self.process_raw_data(src_fname, max_size)
    self.raw_tgt = self.process_raw_data(tgt_fname, max_size)
    assert len(self.raw_src) == len(self.raw_tgt)
    logging.info(f'Filtering data, min len: {min_len}, max len: {max_len}')
    self.filter_raw_data(min_len - 2, max_len - 2)
    assert len(self.raw_src) == len(self.raw_tgt)
    src_lengths = [(i + 2) for i in self.src_len]
    tgt_lengths = [(i + 2) for i in self.tgt_len]
    self.src_lengths = torch.tensor(src_lengths)
    self.tgt_lengths = torch.tensor(tgt_lengths)
    self.lengths = self.src_lengths + self.tgt_lengths
