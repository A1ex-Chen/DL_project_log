def __init__(self, h, d_model, dropout=0.1):
    super().__init__()
    assert d_model % h == 0
    self.d_k = d_model // h
    self.h = h
    self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in
        range(3)])
    self.output_linear = nn.Linear(d_model, d_model)
    self.attention = Attention()
    self.dropout = nn.Dropout(p=dropout)
