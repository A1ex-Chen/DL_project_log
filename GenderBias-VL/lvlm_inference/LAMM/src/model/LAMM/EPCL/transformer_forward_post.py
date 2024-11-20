def forward_post(self, tgt, memory, tgt_mask: Optional[Tensor]=None,
    memory_mask: Optional[Tensor]=None, tgt_key_padding_mask: Optional[
    Tensor]=None, memory_key_padding_mask: Optional[Tensor]=None, pos:
    Optional[Tensor]=None, query_pos: Optional[Tensor]=None,
    return_attn_weights: Optional[bool]=False):
    q = k = self.with_pos_embed(tgt, query_pos)
    tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        key_padding_mask=tgt_key_padding_mask)[0]
    tgt = tgt + self.dropout1(tgt2)
    tgt = self.norm1(tgt)
    tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt,
        query_pos), key=self.with_pos_embed(memory, pos), value=memory,
        attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
    tgt = tgt + self.dropout2(tgt2)
    tgt = self.norm2(tgt)
    tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    tgt = tgt + self.dropout3(tgt2)
    tgt = self.norm3(tgt)
    if return_attn_weights:
        return tgt, attn
    return tgt, None
