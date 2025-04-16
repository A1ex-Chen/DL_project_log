def attention(self, q_x: torch.Tensor, k_x: Optional[torch.Tensor]=None,
    v_x: Optional[torch.Tensor]=None, attn_mask: Optional[torch.Tensor]=None):
    k_x = k_x if k_x is not None else q_x
    v_x = v_x if v_x is not None else q_x
    attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
    return self.attn(q_x, k_x, v_x, attn_mask=attn_mask)
