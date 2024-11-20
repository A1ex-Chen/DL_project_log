def __init__(self, vocab_size=246534, n_positions=256, n_ctx=256, n_embd=
    1280, dff=8192, n_layer=48, n_head=16, resid_pdrop=0.1, embd_pdrop=0.1,
    attn_pdrop=0.1, layer_norm_epsilon=1e-06, initializer_range=0.02,
    summary_type='cls_index', summary_use_proj=True, summary_activation=
    None, summary_proj_to_labels=True, summary_first_dropout=0.1, use_cache
    =True, **kwargs):
    super().__init__(**kwargs)
    self.vocab_size = vocab_size
    self.n_ctx = n_ctx
    self.n_positions = n_positions
    self.n_embd = n_embd
    self.n_layer = n_layer
    self.n_head = n_head
    self.dff = dff
    self.resid_pdrop = resid_pdrop
    self.embd_pdrop = embd_pdrop
    self.attn_pdrop = attn_pdrop
    self.layer_norm_epsilon = layer_norm_epsilon
    self.initializer_range = initializer_range
    self.summary_type = summary_type
    self.summary_use_proj = summary_use_proj
    self.summary_activation = summary_activation
    self.summary_first_dropout = summary_first_dropout
    self.summary_proj_to_labels = summary_proj_to_labels
    self.use_cache = use_cache
