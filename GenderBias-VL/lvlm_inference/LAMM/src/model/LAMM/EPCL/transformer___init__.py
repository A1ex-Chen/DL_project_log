def __init__(self, d_model, nhead=4, dim_feedforward=256, dropout=0.1,
    dropout_attn=None, activation='relu', normalize_before=True,
    norm_fn_name='ln'):
    super().__init__()
    if dropout_attn is None:
        dropout_attn = dropout
    self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
    self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout
        )
    self.norm1 = NORM_DICT[norm_fn_name](d_model)
    self.norm2 = NORM_DICT[norm_fn_name](d_model)
    self.norm3 = NORM_DICT[norm_fn_name](d_model)
    self.dropout1 = nn.Dropout(dropout, inplace=True)
    self.dropout2 = nn.Dropout(dropout, inplace=True)
    self.dropout3 = nn.Dropout(dropout, inplace=True)
    self.linear1 = nn.Linear(d_model, dim_feedforward)
    self.dropout = nn.Dropout(dropout, inplace=True)
    self.linear2 = nn.Linear(dim_feedforward, d_model)
    self.activation = ACTIVATION_DICT[activation]()
    self.normalize_before = normalize_before
