def get_sent_stream(self, path):
    sents = self.vocab.encode_file(path, add_double_eos=True)
    if self.shuffle:
        np.random.shuffle(sents)
    sent_stream = iter(sents)
    return sent_stream
