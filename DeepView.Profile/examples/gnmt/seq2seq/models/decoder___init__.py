def __init__(self, vocab_size, hidden_size=1024, num_layers=4, dropout=0.2,
    batch_first=False, embedder=None, init_weight=0.1):
    """
        Constructor of the ResidualRecurrentDecoder.

        :param vocab_size: size of vocabulary
        :param hidden_size: hidden size for LSMT layers
        :param num_layers: number of LSTM layers
        :param dropout: probability of dropout (on input to LSTM layers)
        :param batch_first: if True the model uses (batch,seq,feature) tensors,
            if false the model uses (seq, batch, feature)
        :param embedder: instance of nn.Embedding, if None constructor will
            create new embedding layer
        :param init_weight: range for the uniform initializer
        """
    super(ResidualRecurrentDecoder, self).__init__()
    self.num_layers = num_layers
    self.att_rnn = RecurrentAttention(hidden_size, hidden_size, hidden_size,
        num_layers=1, batch_first=batch_first, dropout=dropout)
    self.rnn_layers = nn.ModuleList()
    for _ in range(num_layers - 1):
        self.rnn_layers.append(nn.LSTM(2 * hidden_size, hidden_size,
            num_layers=1, bias=True, batch_first=batch_first))
    for lstm in self.rnn_layers:
        init_lstm_(lstm, init_weight)
    if embedder is not None:
        self.embedder = embedder
    else:
        self.embedder = nn.Embedding(vocab_size, hidden_size, padding_idx=
            config.PAD)
        nn.init.uniform_(self.embedder.weight.data, -init_weight, init_weight)
    self.classifier = Classifier(hidden_size, vocab_size)
    self.dropout = nn.Dropout(p=dropout)
