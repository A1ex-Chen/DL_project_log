def __init__(self, vocab_size_or_config_json_file=30522, hidden_size=768,
    num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072,
    hidden_act='gelu', hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1, max_position_embeddings=512,
    type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, **kwargs):
    super(BertConfig, self).__init__(**kwargs)
    if isinstance(vocab_size_or_config_json_file, str) or sys.version_info[0
        ] == 2 and isinstance(vocab_size_or_config_json_file, unicode):
        with open(vocab_size_or_config_json_file, 'r', encoding='utf-8'
            ) as reader:
            json_config = json.loads(reader.read())
        for key, value in json_config.items():
            self.__dict__[key] = value
    elif isinstance(vocab_size_or_config_json_file, int):
        self.vocab_size = vocab_size_or_config_json_file
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
    else:
        raise ValueError(
            'First argument must be either a vocabulary size (int) or the path to a pretrained model config file (str)'
            )
