def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]=None):
    return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
