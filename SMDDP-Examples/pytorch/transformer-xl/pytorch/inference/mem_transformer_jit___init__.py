def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
    dropout, dropatt, dtype, tie_weight=True, d_embed=None, div_val=1,
    tie_projs=[False], pre_lnorm=False, tgt_len=None, ext_len=None, mem_len
    =None, cutoffs=[], adapt_inp=False, same_length=False, attn_type=0,
    clamp_len=-1, sample_softmax=-1):
    super(MemTransformerLM, self).__init__()
    self.n_token = n_token
    d_embed = d_model if d_embed is None else d_embed
    self.d_embed = d_embed
    self.d_model = d_model
    self.n_head = n_head
    self.d_head = d_head
    self.dtype = dtype
    self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs,
        div_val=div_val, dtype=dtype)
    self.drop = nn.Dropout(dropout)
    self.tie_weight = tie_weight
    self.tie_projs = tie_projs
    self.div_val = div_val
    self.n_layer = n_layer
    self.tgt_len = tgt_len
    self.mem_len = mem_len
    self.ext_len = ext_len
    self.max_klen = tgt_len + ext_len + mem_len
    self.attn_type = attn_type
    if attn_type != 0:
        raise RuntimeError('TorchScripted model supports only attn_type == 0')
    self.layers = nn.ModuleList()
    if attn_type == 0:
        for i in range(n_layer):
            self.layers.append(RelPartialLearnableDecoderLayer(n_head,
                d_model, d_head, d_inner, dropout, tgt_len=tgt_len, ext_len
                =ext_len, mem_len=mem_len, dropatt=dropatt, pre_lnorm=
                pre_lnorm))
    self.sample_softmax = sample_softmax
    if sample_softmax > 0:
        self.out_layer = nn.Linear(d_model, n_token)
        self.tie_weight = tie_weight
        self.sampler = LogUniformSampler(n_token, sample_softmax)
    else:
        if tie_weight:
            emb_layers = [i.weight for i in self.word_emb.emb_layers]
        else:
            emb_layers = None
        emb_projs = self.word_emb.emb_projs
        self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model,
            cutoffs, div_val=div_val, dtype=dtype, tie_projs=tie_projs,
            out_projs=emb_projs, out_layers_weights=emb_layers)
    self.same_length = same_length
    self.clamp_len = clamp_len
    self._create_params()
