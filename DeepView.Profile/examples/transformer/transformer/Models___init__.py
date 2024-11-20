def __init__(self, n_src_vocab, n_tgt_vocab, len_max_seq, d_word_vec=512,
    d_model=512, d_inner=2048, n_layers=6, n_head=8, d_k=64, d_v=64,
    dropout=0.1, tgt_emb_prj_weight_sharing=True,
    emb_src_tgt_weight_sharing=True):
    super().__init__()
    self.encoder = Encoder(n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
        d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner, n_layers=
        n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)
    self.decoder = Decoder(n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
        d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner, n_layers=
        n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)
    self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
    nn.init.xavier_normal_(self.tgt_word_prj.weight)
    assert d_model == d_word_vec, 'To facilitate the residual connections,          the dimensions of all module outputs shall be the same.'
    self.tgt_emb_prj_weight_sharing = tgt_emb_prj_weight_sharing
    if self.tgt_emb_prj_weight_sharing:
        self.x_logit_scale = d_model ** -0.5
    else:
        self.x_logit_scale = 1.0
    if emb_src_tgt_weight_sharing:
        assert n_src_vocab == n_tgt_vocab, 'To share word embedding table, the vocabulary size of src/tgt shall be the same.'
        self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight
