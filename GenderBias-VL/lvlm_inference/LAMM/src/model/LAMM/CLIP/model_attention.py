def attention(self, x: torch.Tensor):
    self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device
        ) if self.attn_mask is not None else None
    return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
