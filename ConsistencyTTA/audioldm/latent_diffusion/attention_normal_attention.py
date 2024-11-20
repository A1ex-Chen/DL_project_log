def normal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """
        #### Normal Attention

        :param q: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param k: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param v: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        """
    q = q.view(*q.shape[:2], self.n_heads, -1)
    k = k.view(*k.shape[:2], self.n_heads, -1)
    v = v.view(*v.shape[:2], self.n_heads, -1)
    attn = torch.einsum('bihd,bjhd->bhij', q, k) * self.scale
    if self.is_inplace:
        half = attn.shape[0] // 2
        attn[half:] = attn[half:].softmax(dim=-1)
        attn[:half] = attn[:half].softmax(dim=-1)
    else:
        attn = attn.softmax(dim=-1)
    out = torch.einsum('bhij,bjhd->bihd', attn, v)
    out = out.reshape(*out.shape[:2], -1)
    return self.to_out(out)
