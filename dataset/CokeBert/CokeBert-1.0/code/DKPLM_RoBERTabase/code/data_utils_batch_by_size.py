def batch_by_size(indices, num_tokens_fn, max_tokens=None, max_sentences=
    None, required_batch_size_multiple=1):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch.
            Default: ``None``
        max_sentences (int, optional): max number of sentences in each
            batch. Default: ``None``
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N. Default: ``1``
    """
    max_tokens = max_tokens if max_tokens is not None else float('Inf')
    max_sentences = max_sentences if max_sentences is not None else float('Inf'
        )
    bsz_mult = required_batch_size_multiple
    batch = []

    def is_batch_full(num_tokens):
        if len(batch) == 0:
            return False
        if len(batch) == max_sentences:
            return True
        if num_tokens > max_tokens:
            return True
        return False
    sample_len = 0
    sample_lens = []
    for idx in indices:
        sample_lens.append(num_tokens_fn(idx))
        sample_len = max(sample_len, sample_lens[-1])
        num_tokens = (len(batch) + 1) * sample_len
        if is_batch_full(num_tokens):
            mod_len = max(bsz_mult * (len(batch) // bsz_mult), len(batch) %
                bsz_mult)
            yield batch[:mod_len]
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
        batch.append(idx)
    if len(batch) > 0:
        yield batch
