def __init__(self, config):
    super().__init__()
    assert config.vocab_size is not None
    assert config.block_size is not None
    self.config = config
    self.transformer = nn.ModuleDict(dict(wte=nn.Embedding(config.
        vocab_size, config.n_embd), wpe=nn.Embedding(config.block_size,
        config.n_embd), drop=nn.Dropout(config.dropout), h=nn.ModuleList([
        Block(config) for _ in range(config.n_layer)]), ln_f=LayerNorm(
        config.n_embd, bias=config.bias)))
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    self.transformer.wte.weight = self.lm_head.weight
    self.apply(self._init_weights)
    for pn, p in self.named_parameters():
        if pn.endswith('c_proj.weight'):
            torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 *
                config.n_layer))
    print('number of parameters: %.2fM' % (self.get_num_params() / 1000000.0,))
