def __init__(self, vocab_size_or_config_json_file=30145, emb_dim=2048,
    n_layers=12, n_heads=16, dropout=0.1, attention_dropout=0.1,
    gelu_activation=True, sinusoidal_embeddings=False, causal=False, asm=
    False, n_langs=1, use_lang_emb=True, max_position_embeddings=512,
    embed_init_std=2048 ** -0.5, layer_norm_eps=1e-12, init_std=0.02,
    bos_index=0, eos_index=1, pad_index=2, unk_index=3, mask_index=5,
    is_encoder=True, finetuning_task=None, num_labels=2, summary_type=
    'first', summary_use_proj=True, summary_activation=None,
    summary_proj_to_labels=True, summary_first_dropout=0.1, start_n_top=5,
    end_n_top=5, **kwargs):
    """Constructs XLMConfig.
        """
    super(XLMConfig, self).__init__(**kwargs)
    if isinstance(vocab_size_or_config_json_file, str) or sys.version_info[0
        ] == 2 and isinstance(vocab_size_or_config_json_file, unicode):
        with open(vocab_size_or_config_json_file, 'r', encoding='utf-8'
            ) as reader:
            json_config = json.loads(reader.read())
        for key, value in json_config.items():
            self.__dict__[key] = value
    elif isinstance(vocab_size_or_config_json_file, int):
        self.n_words = vocab_size_or_config_json_file
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.gelu_activation = gelu_activation
        self.sinusoidal_embeddings = sinusoidal_embeddings
        self.causal = causal
        self.asm = asm
        self.n_langs = n_langs
        self.use_lang_emb = use_lang_emb
        self.layer_norm_eps = layer_norm_eps
        self.bos_index = bos_index
        self.eos_index = eos_index
        self.pad_index = pad_index
        self.unk_index = unk_index
        self.mask_index = mask_index
        self.is_encoder = is_encoder
        self.max_position_embeddings = max_position_embeddings
        self.embed_init_std = embed_init_std
        self.init_std = init_std
        self.finetuning_task = finetuning_task
        self.num_labels = num_labels
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_proj_to_labels = summary_proj_to_labels
        self.summary_first_dropout = summary_first_dropout
        self.start_n_top = start_n_top
        self.end_n_top = end_n_top
    else:
        raise ValueError(
            'First argument must be either a vocabulary size (int) or the path to a pretrained model config file (str)'
            )
