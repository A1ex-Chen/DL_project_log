def forward(self, tgt, memory, tgt_mask: Optional[Tensor]=None, memory_mask:
    Optional[Tensor]=None, tgt_key_padding_mask: Optional[Tensor]=None,
    memory_key_padding_mask: Optional[Tensor]=None, pos: Optional[Tensor]=
    None, query_pos: Optional[Tensor]=None):
    if self.normalize_before:
        return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
            tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
    return self.forward_post(tgt, memory, tgt_mask, memory_mask,
        tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
