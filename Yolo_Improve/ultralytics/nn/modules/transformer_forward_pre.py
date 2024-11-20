def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
    """Performs forward pass with pre-normalization."""
    src2 = self.norm1(src)
    q = k = self.with_pos_embed(src2, pos)
    src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=
        src_key_padding_mask)[0]
    src = src + self.dropout1(src2)
    src2 = self.norm2(src)
    src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
    return src + self.dropout2(src2)
