def __init__(self, vocab_size_or_config_json_file=30522,
    max_position_embeddings=512, sinusoidal_pos_embds=False, n_layers=6,
    n_heads=12, dim=768, hidden_dim=4 * 768, dropout=0.1, attention_dropout
    =0.1, activation='gelu', initializer_range=0.02, tie_weights_=True,
    qa_dropout=0.1, seq_classif_dropout=0.2, **kwargs):
    super(DistilBertConfig, self).__init__(**kwargs)
    if isinstance(vocab_size_or_config_json_file, str) or sys.version_info[0
        ] == 2 and isinstance(vocab_size_or_config_json_file, unicode):
        with open(vocab_size_or_config_json_file, 'r', encoding='utf-8'
            ) as reader:
            json_config = json.loads(reader.read())
        for key, value in json_config.items():
            self.__dict__[key] = value
    elif isinstance(vocab_size_or_config_json_file, int):
        self.vocab_size = vocab_size_or_config_json_file
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
        self.tie_weights_ = tie_weights_
        self.qa_dropout = qa_dropout
        self.seq_classif_dropout = seq_classif_dropout
    else:
        raise ValueError(
            'First argument must be either a vocabulary size (int) or the path to a pretrained model config file (str)'
            )
