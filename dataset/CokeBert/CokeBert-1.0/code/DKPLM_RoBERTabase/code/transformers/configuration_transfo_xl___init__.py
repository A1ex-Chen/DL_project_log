def __init__(self, vocab_size_or_config_json_file=267735, cutoffs=[20000, 
    40000, 200000], d_model=1024, d_embed=1024, n_head=16, d_head=64,
    d_inner=4096, div_val=4, pre_lnorm=False, n_layer=18, tgt_len=128,
    ext_len=0, mem_len=1600, clamp_len=1000, same_length=True,
    proj_share_all_but_first=True, attn_type=0, sample_softmax=-1, adaptive
    =True, tie_weight=True, dropout=0.1, dropatt=0.0, untie_r=True, init=
    'normal', init_range=0.01, proj_init_std=0.01, init_std=0.02,
    layer_norm_epsilon=1e-05, **kwargs):
    """Constructs TransfoXLConfig.
        """
    super(TransfoXLConfig, self).__init__(**kwargs)
    self.n_token = vocab_size_or_config_json_file if isinstance(
        vocab_size_or_config_json_file, int) else -1
    self.cutoffs = []
    self.cutoffs.extend(cutoffs)
    self.tie_weight = tie_weight
    if proj_share_all_but_first:
        self.tie_projs = [False] + [True] * len(self.cutoffs)
    else:
        self.tie_projs = [False] + [False] * len(self.cutoffs)
    self.d_model = d_model
    self.d_embed = d_embed
    self.d_head = d_head
    self.d_inner = d_inner
    self.div_val = div_val
    self.pre_lnorm = pre_lnorm
    self.n_layer = n_layer
    self.n_head = n_head
    self.tgt_len = tgt_len
    self.ext_len = ext_len
    self.mem_len = mem_len
    self.same_length = same_length
    self.attn_type = attn_type
    self.clamp_len = clamp_len
    self.sample_softmax = sample_softmax
    self.adaptive = adaptive
    self.dropout = dropout
    self.dropatt = dropatt
    self.untie_r = untie_r
    self.init = init
    self.init_range = init_range
    self.proj_init_std = proj_init_std
    self.init_std = init_std
    self.layer_norm_epsilon = layer_norm_epsilon
    if isinstance(vocab_size_or_config_json_file, str) or sys.version_info[0
        ] == 2 and isinstance(vocab_size_or_config_json_file, unicode):
        with open(vocab_size_or_config_json_file, 'r', encoding='utf-8'
            ) as reader:
            json_config = json.loads(reader.read())
        for key, value in json_config.items():
            self.__dict__[key] = value
    elif not isinstance(vocab_size_or_config_json_file, int):
        raise ValueError(
            'First argument must be either a vocabulary size (int) or the path to a pretrained model config file (str)'
            )
