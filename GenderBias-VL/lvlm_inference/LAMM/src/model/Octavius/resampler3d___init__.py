def __init__(self, num_query, num_heads, num_layers, hidden_size):
    super().__init__()
    self.num_query = num_query
    self.num_heads = num_heads
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.query_embed = nn.Embedding(self.num_query, self.hidden_size)
    self.cross_attn = nn.ModuleList(MultiHeadAttention(d_model=hidden_size,
        d_k=hidden_size // num_heads, d_v=hidden_size // num_heads, h=
        num_heads) for _ in range(num_layers))
    self.self_attn = nn.ModuleList(MultiHeadAttention(d_model=hidden_size,
        d_k=hidden_size // num_heads, d_v=hidden_size // num_heads, h=
        num_heads) for _ in range(self.num_layers))
