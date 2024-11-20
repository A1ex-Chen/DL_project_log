def __init__(self, *args, **kwargs):
    self.vocab = TransfoXLTokenizer(*args, **kwargs)
    self.dataset = None
    self.train = None
    self.valid = None
    self.test = None
