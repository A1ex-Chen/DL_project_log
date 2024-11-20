def get_batch(context_tokens: torch.LongTensor, eod_id: int):
    """Generate batch from context tokens."""
    tokens = context_tokens.contiguous().to(context_tokens.device)
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(tokens,
        eod_id, reset_position_ids=False, reset_attention_mask=False,
        eod_mask_loss=False)
    return tokens, attention_mask, position_ids
