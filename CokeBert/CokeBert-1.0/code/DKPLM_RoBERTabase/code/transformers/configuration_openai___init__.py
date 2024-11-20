def __init__(self, vocab_size_or_config_json_file=40478, n_positions=512,
    n_ctx=512, n_embd=768, n_layer=12, n_head=12, afn='gelu', resid_pdrop=
    0.1, embd_pdrop=0.1, attn_pdrop=0.1, layer_norm_epsilon=1e-05,
    initializer_range=0.02, predict_special_tokens=True, num_labels=1,
    summary_type='cls_index', summary_use_proj=True, summary_activation=
    None, summary_proj_to_labels=True, summary_first_dropout=0.1, **kwargs):
    """Constructs OpenAIGPTConfig.
        """
    super(OpenAIGPTConfig, self).__init__(**kwargs)
    if isinstance(vocab_size_or_config_json_file, str) or sys.version_info[0
        ] == 2 and isinstance(vocab_size_or_config_json_file, unicode):
        with open(vocab_size_or_config_json_file, 'r', encoding='utf-8'
            ) as reader:
            json_config = json.loads(reader.read())
        for key, value in json_config.items():
            self.__dict__[key] = value
    elif isinstance(vocab_size_or_config_json_file, int):
        self.vocab_size = vocab_size_or_config_json_file
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.afn = afn
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.predict_special_tokens = predict_special_tokens
        self.num_labels = num_labels
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
    else:
        raise ValueError(
            'First argument must be either a vocabulary size (int)or the path to a pretrained model config file (str)'
            )
