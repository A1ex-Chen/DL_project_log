def is_batch_full(num_tokens):
    if len(batch) == 0:
        return False
    if len(batch) == max_sentences:
        return True
    if num_tokens > max_tokens:
        return True
    return False
