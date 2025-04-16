def add_decomposed_rel_pos(attn: torch.Tensor, q: torch.Tensor, rel_pos_h:
    torch.Tensor, rel_pos_w: torch.Tensor, q_size: Tuple[int, int], k_size:
    Tuple[int, int]) ->torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from mvitv2 paper at
    https://github.com/facebookresearch/mvit/blob/main/mvit/models/attention.py.

    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum('bhwc,hkc->bhwk', r_q, Rh)
    rel_w = torch.einsum('bhwc,wkc->bhwk', r_q, Rw)
    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] +
        rel_w[:, :, :, None, :]).view(B, q_h * q_w, k_h * k_w)
    return attn
