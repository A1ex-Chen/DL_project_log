def __init__(self, params: ModelArgs):
    super().__init__()
    self.params = params
    self.vocab_size = params.vocab_size
    self.n_layers = params.n_layers
    self.tok_embeddings = Embedding(params.vocab_size, params.dim)
    self.layers = torch.nn.ModuleList()
    for layer_id in range(params.n_layers):
        self.layers.append(TransformerBlock(layer_id, params))
    self.norm = RMSNorm(params.dim, eps=params.norm_eps)
    self.output = Linear(params.dim, params.vocab_size, bias=False)
    self.freqs_cis = precompute_freqs_cis(self.params.dim // self.params.
        n_heads, self.params.max_seq_len * 2)
