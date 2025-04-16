def get_masks(slen, lengths, causal, padding_mask=None):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    if padding_mask is not None:
        mask = padding_mask
    else:
        assert lengths.max().item() <= slen
        mask = alen < lengths[:, None]
    bs = lengths.size(0)
    if causal:
        attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :,
            None]
    else:
        attn_mask = mask
    assert mask.size() == (bs, slen)
    assert causal is False or attn_mask.size() == (bs, slen, slen)
    return mask, attn_mask
