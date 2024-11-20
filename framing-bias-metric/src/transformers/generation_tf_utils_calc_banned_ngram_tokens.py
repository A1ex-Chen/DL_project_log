def calc_banned_ngram_tokens(prev_input_ids, num_hypos,
    no_repeat_ngram_size, cur_len):
    if cur_len + 1 < no_repeat_ngram_size:
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].numpy().tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]
            ):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(
                prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].numpy
            ().tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])
    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(
        num_hypos)]
    return banned_tokens
