def scaled_dot_product_gqa(query: Tensor, key: Tensor, value: Tensor,
    dropout: float=0.0, scale: Optional[float]=None, mask: Optional[Tensor]
    =None, is_causal: Optional[bool]=None, need_weights: bool=False,
    average_attn_weights: bool=False, force_grouped: bool=False):
    """Scaled dot product attention with support for grouped queries.

    Einstein notation:
    - b: batch size
    - n / s: sequence length
    - h: number of heads
    - g: number of groups
    - d: dimension of query/key/value

    Args:
        query: Query tensor of shape (b, n, h, d)
        key: Key tensor of shape (b, s, h, d)
        value: Value tensor of shape (b, s, h, d)
        dropout: Dropout probability (default: 0.0)
        scale: Scale factor for query (default: d_query ** 0.5)
        mask: Mask tensor of shape (b, n, s) or (b, s). If 'ndim == 2', the mask is
            applied to all 'n' rows of the attention matrix. (default: None)
        force_grouped: If True, apply grouped-query attention even if the number of
            heads is equal for query, key, and value. (default: False)

    Returns:
        2-tuple of:
        - Attention output with shape (b, n, h, d)
        - (Optional) Attention weights with shape (b, h, n, s). Only returned if
          'need_weights' is True.
    """
    if mask is not None and is_causal is not None:
        raise ValueError(
            "Only one of 'mask' and 'is_causal' should be provided, but got both."
            )
    elif not query.ndim == key.ndim == value.ndim == 4:
        raise ValueError(
            f'Expected query, key, and value to be 4-dimensional, but got shapes {query.shape}, {key.shape}, and {value.shape}.'
            )
    query = rearrange(query, 'b n h d -> b h n d')
    key = rearrange(key, 'b s h d -> b h s d')
    value = rearrange(value, 'b s h d -> b h s d')
    bq, hq, nq, dq = query.shape
    bk, hk, nk, dk = key.shape
    bv, hv, nv, dv = value.shape
    if not (bq == bk == bv and dq == dk == dv):
        raise ValueError(
            f'Expected query, key, and value to have the same batch size (dim=0) and embedding dimension (dim=3), but got query: {query.shape}, key: {key.shape}, and value: {value.shape}.'
            )
    elif hk != hv or nk != nv:
        raise ValueError(
            f'Expected key and value to have the same size in dimensions 1 and 2, but got key: {key.shape} and value: {value.shape}.'
            )
    elif hq % hk != 0:
        raise ValueError(
            f'Expected query heads to be a multiple of key/value heads, but got query: {query.shape} and key/value: {key.shape}.'
            )
    if scale is None:
        scale = query.size(-1) ** 0.5
    query = query / scale
    num_head_groups = hq // hk
    if num_head_groups > 1 or force_grouped:
        query = rearrange(query, 'b (h g) n d -> b g h n d', g=num_head_groups)
        similarity = einsum(query, key, 'b g h n d, b h s d -> b h n s')
    else:
        similarity = einsum(query, key, 'b h n d, b h s d -> b h n s')
    if is_causal:
        mask = torch.ones((bq, nq, nk), device=query.device, dtype=torch.bool
            ).tril_()
    if mask is not None:
        if mask.ndim == 2:
            mask = rearrange(mask, 'b s -> b () () s')
        elif mask.ndim == 3:
            mask = rearrange(mask, 'b n s -> b () n s')
        similarity.masked_fill_(~mask, torch.finfo(similarity.dtype).min)
    attention = F.softmax(similarity / scale, dim=-1)
    if dropout > 0.0:
        attention = F.dropout(attention, p=dropout)
    out = einsum(attention, value, 'b h n s, b h s d -> b h n d')
    out = rearrange(out, 'b h n d -> b n h d')
    attn_weights: Optional[Tensor] = None
    if need_weights:
        attn_weights = rearrange(attention, 'b h n s -> b n s h')
        if average_attn_weights:
            attn_weights = attn_weights.mean(dim=1)
    return out, attn_weights
