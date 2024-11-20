def __init__(self, config: InternLM2Config):
    super().__init__(config)
    self.padding_idx = config.pad_token_id
    self.vocab_size = config.vocab_size
    self.tok_embeddings = nn.Embedding(config.vocab_size, config.
        hidden_size, self.padding_idx)
    self.layers = nn.ModuleList([InternLM2DecoderLayer(config) for _ in
        range(config.num_hidden_layers)])
    self.norm = InternLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.gradient_checkpointing = False
    self.post_init()
