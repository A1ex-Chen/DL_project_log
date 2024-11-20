def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
    super(DecoderLayer, self).__init__()
    self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=
        dropout)
    self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=
        dropout)
    self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
