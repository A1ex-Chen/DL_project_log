def __init__(self, vocab_size=30522, max_position_embeddings=512,
    sinusoidal_pos_embds=False, n_layers=6, n_heads=12, dim=768, hidden_dim
    =4 * 768, dropout=0.1, attention_dropout=0.1, activation='gelu',
    initializer_range=0.02, qa_dropout=0.1, seq_classif_dropout=0.2,
    pad_token_id=0, **kwargs):
    super().__init__(**kwargs, pad_token_id=pad_token_id)
    self.vocab_size = vocab_size
    self.max_position_embeddings = max_position_embeddings
    self.sinusoidal_pos_embds = sinusoidal_pos_embds
    self.n_layers = n_layers
    self.n_heads = n_heads
    self.dim = dim
    self.hidden_dim = hidden_dim
    self.dropout = dropout
    self.attention_dropout = attention_dropout
    self.activation = activation
    self.initializer_range = initializer_range
    self.qa_dropout = qa_dropout
    self.seq_classif_dropout = seq_classif_dropout
