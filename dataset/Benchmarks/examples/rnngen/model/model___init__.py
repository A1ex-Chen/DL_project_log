def __init__(self, vocab_size, emb_size, max_len=320):
    super(CharRNN, self).__init__()
    self.max_len = max_len
    self.emb = nn.Embedding(vocab_size, emb_size)
    self.lstm = nn.LSTM(emb_size, 256, dropout=0.1, num_layers=4)
    self.linear = nn.Linear(256, vocab_size)
