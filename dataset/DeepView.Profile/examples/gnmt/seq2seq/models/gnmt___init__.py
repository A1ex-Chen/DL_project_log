def __init__(self, vocab_size, hidden_size=1024, num_layers=4, dropout=0.2,
    batch_first=False, share_embedding=True):
    """
        Constructor for the GNMT v2 model.

        :param vocab_size: size of vocabulary (number of tokens)
        :param hidden_size: internal hidden size of the model
        :param num_layers: number of layers, applies to both encoder and
            decoder
        :param dropout: probability of dropout (in encoder and decoder)
        :param batch_first: if True the model uses (batch,seq,feature) tensors,
            if false the model uses (seq, batch, feature)
        :param share_embedding: if True embeddings are shared between encoder
            and decoder
        """
    super(GNMT, self).__init__(batch_first=batch_first)
    if share_embedding:
        embedder = nn.Embedding(vocab_size, hidden_size, padding_idx=config.PAD
            )
        nn.init.uniform_(embedder.weight.data, -0.1, 0.1)
    else:
        embedder = None
    self.encoder = ResidualRecurrentEncoder(vocab_size, hidden_size,
        num_layers, dropout, batch_first, embedder)
    self.decoder = ResidualRecurrentDecoder(vocab_size, hidden_size,
        num_layers, dropout, batch_first, embedder)
