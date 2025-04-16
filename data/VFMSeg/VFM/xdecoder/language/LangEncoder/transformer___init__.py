def __init__(self, context_length: int, vocab_size: int, width: int, layers:
    int, heads: int, drop_path: float=0.0, autogressive: bool=True):
    super().__init__()
    self.token_embedding = nn.Embedding(vocab_size, width)
    self.context_length = context_length
    self.positional_embedding = nn.Parameter(torch.empty(self.
        context_length, width))
    self.width = width
    self.layers = layers
    self.autogressive = autogressive
    attn_mask = self.build_attention_mask() if autogressive else None
    dpr = [x.item() for x in torch.linspace(0, drop_path, layers)]
    self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads,
        attn_mask, dpr[i]) for i in range(layers)])
    self.ln_final = LayerNorm(width)
    trunc_normal_(self.positional_embedding, std=0.02)
    trunc_normal_(self.token_embedding.weight, std=0.02)
    self.apply(self._init_weights)
