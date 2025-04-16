def __init__(self, max_len, d_model):
    super().__init__()
    self.pe = nn.Embedding(max_len, d_model)
