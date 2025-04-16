@register_to_config
def __init__(self, max_length: int, vocab_size: int, d_model: int,
    dropout_rate: float, num_layers: int, num_heads: int, d_kv: int, d_ff:
    int, feed_forward_proj: str, is_decoder: bool=False):
    super().__init__()
    self.token_embedder = nn.Embedding(vocab_size, d_model)
    self.position_encoding = nn.Embedding(max_length, d_model)
    self.position_encoding.weight.requires_grad = False
    self.dropout_pre = nn.Dropout(p=dropout_rate)
    t5config = T5Config(vocab_size=vocab_size, d_model=d_model, num_heads=
        num_heads, d_kv=d_kv, d_ff=d_ff, dropout_rate=dropout_rate,
        feed_forward_proj=feed_forward_proj, is_decoder=is_decoder,
        is_encoder_decoder=False)
    self.encoders = nn.ModuleList()
    for lyr_num in range(num_layers):
        lyr = T5Block(t5config)
        self.encoders.append(lyr)
    self.layer_norm = T5LayerNorm(d_model)
    self.dropout_post = nn.Dropout(p=dropout_rate)
