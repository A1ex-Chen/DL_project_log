def _get_generated_ngrams(hypo_idx):
    start_idx = cur_len + 1 - self.ngram_size
    ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
    return generated_ngrams[hypo_idx].get(ngram_idx, [])
