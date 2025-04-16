def repeat_kv(x: torch.Tensor, n_rep: int) ->torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(bs, slen, n_kv_heads, n_rep, head_dim
        ).reshape(bs, slen, n_kv_heads * n_rep, head_dim)
