def __init__(self, config: CLIPTextConfig):
    super().__init__()
    embed_dim = config.hidden_size
    self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
    self.position_embedding = nn.Embedding(config.max_position_embeddings,
        embed_dim)
    self.register_buffer('position_ids', torch.arange(config.
        max_position_embeddings).expand((1, -1)))
