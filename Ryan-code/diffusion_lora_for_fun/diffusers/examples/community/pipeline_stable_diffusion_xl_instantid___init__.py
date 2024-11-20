def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0,
    num_tokens=4):
    super().__init__()
    self.hidden_size = hidden_size
    self.cross_attention_dim = cross_attention_dim
    self.scale = scale
    self.num_tokens = num_tokens
    self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size,
        hidden_size, bias=False)
    self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size,
        hidden_size, bias=False)
