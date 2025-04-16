def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
    """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """
    super().__init__()
    self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden,
        dropout=dropout)
    self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=
        feed_forward_hidden, dropout=dropout)
    self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
    self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
    self.dropout = nn.Dropout(p=dropout)
