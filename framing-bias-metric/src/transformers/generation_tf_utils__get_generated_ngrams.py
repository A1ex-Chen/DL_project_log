def _get_generated_ngrams(hypo_idx):
    start_idx = cur_len + 1 - no_repeat_ngram_size
    ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].numpy().
        tolist())
    return generated_ngrams[hypo_idx].get(ngram_idx, [])
