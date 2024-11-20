def adapt_tokenizer_for_denoising(tokenizer: Tokenizer):
    """Adds sentinel tokens and padding token (if missing).

    Expands the tokenizer vocabulary to include sentinel tokens
    used in mixture-of-denoiser tasks as well as a padding token.

    All added tokens are added as special tokens. No tokens are
    added if sentinel tokens and padding token already exist.
    """
    sentinels_to_add = [f'<extra_id_{i}>' for i in range(NUM_SENTINEL_TOKENS)]
    tokenizer.add_tokens(sentinels_to_add, special_tokens=True)
    if tokenizer.pad_token is None:
        tokenizer.add_tokens('<pad>', special_tokens=True)
        tokenizer.pad_token = '<pad>'
        assert tokenizer.pad_token_id is not None
    sentinels = ''.join([f'<extra_id_{i}>' for i in range(NUM_SENTINEL_TOKENS)]
        )
    _sentinel_token_ids = tokenizer(sentinels, add_special_tokens=False
        ).input_ids
    tokenizer.sentinel_token_ids = _sentinel_token_ids
