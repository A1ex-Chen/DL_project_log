def __init__(self, d_in, d_hid, dropout=0.1):
    super().__init__()
    self.w_1 = nn.Conv1d(d_in, d_hid, 1)
    self.w_2 = nn.Conv1d(d_hid, d_in, 1)
    self.layer_norm = nn.LayerNorm(d_in)
    self.dropout = nn.Dropout(dropout)
