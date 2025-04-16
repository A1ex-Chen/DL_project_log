def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None
    ):
    """Performs forward pass with post-normalization."""
    q = k = self.with_pos_embed(src, pos)
    src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=
        src_key_padding_mask)[0]
    src = src + self.dropout1(src2)
    src = self.norm1(src)
    src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
    src = src + self.dropout2(src2)
    return self.norm2(src)
