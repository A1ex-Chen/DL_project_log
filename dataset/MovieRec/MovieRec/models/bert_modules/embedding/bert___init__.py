def __init__(self, vocab_size, embed_size, max_len, dropout=0.1):
    """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
    super().__init__()
    self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
    self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)
    self.dropout = nn.Dropout(p=dropout)
    self.embed_size = embed_size
