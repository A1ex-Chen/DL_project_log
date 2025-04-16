def attention(self, x: torch.Tensor, key_padding_mask: torch.Tensor=None):
    self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device
        ) if self.attn_mask is not None else None
    return self.attn(x, x, x, key_padding_mask=key_padding_mask,
        need_weights=False, attn_mask=self.attn_mask)[0]
