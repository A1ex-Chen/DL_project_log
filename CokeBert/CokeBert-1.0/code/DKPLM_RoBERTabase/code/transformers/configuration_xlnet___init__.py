def __init__(self, vocab_size_or_config_json_file=32000, d_model=1024,
    n_layer=24, n_head=16, d_inner=4096, max_position_embeddings=512,
    ff_activation='gelu', untie_r=True, attn_type='bi', initializer_range=
    0.02, layer_norm_eps=1e-12, dropout=0.1, mem_len=None, reuse_len=None,
    bi_data=False, clamp_len=-1, same_length=False, finetuning_task=None,
    num_labels=2, summary_type='last', summary_use_proj=True,
    summary_activation='tanh', summary_last_dropout=0.1, start_n_top=5,
    end_n_top=5, **kwargs):
    """Constructs XLNetConfig.
        """
    super(XLNetConfig, self).__init__(**kwargs)
    if isinstance(vocab_size_or_config_json_file, str) or sys.version_info[0
        ] == 2 and isinstance(vocab_size_or_config_json_file, unicode):
        with open(vocab_size_or_config_json_file, 'r', encoding='utf-8'
            ) as reader:
            json_config = json.loads(reader.read())
        for key, value in json_config.items():
            setattr(config, key, value)
    elif isinstance(vocab_size_or_config_json_file, int):
        self.n_token = vocab_size_or_config_json_file
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        assert d_model % n_head == 0
        self.d_head = d_model // n_head
        self.ff_activation = ff_activation
        self.d_inner = d_inner
        self.untie_r = untie_r
        self.attn_type = attn_type
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.dropout = dropout
        self.mem_len = mem_len
        self.reuse_len = reuse_len
        self.bi_data = bi_data
        self.clamp_len = clamp_len
        self.same_length = same_length
        self.finetuning_task = finetuning_task
        self.num_labels = num_labels
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_last_dropout = summary_last_dropout
        self.start_n_top = start_n_top
        self.end_n_top = end_n_top
    else:
        raise ValueError(
            'First argument must be either a vocabulary size (int) or the path to a pretrained model config file (str)'
            )
