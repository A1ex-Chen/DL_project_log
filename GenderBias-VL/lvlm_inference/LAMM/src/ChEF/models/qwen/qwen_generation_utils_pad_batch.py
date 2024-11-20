def pad_batch(batch: BatchTokensType, pad_id: int, seq_length: int
    ) ->BatchTokensType:
    for tokens in batch:
        context_length = len(tokens)
        if context_length < seq_length:
            tokens.extend([pad_id] * (seq_length - context_length))
    return batch
