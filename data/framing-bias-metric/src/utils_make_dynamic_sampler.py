def make_dynamic_sampler(self, max_tokens_per_batch=1024, **kwargs):
    assert FAIRSEQ_AVAILABLE, 'Dynamic batch size requires `pip install fairseq`'
    assert not self.used_char_len, 'You must call  python make_len_file.py before calling make_dynamic_sampler'
    sorted_indices = list(self.make_sortish_sampler(1024, shuffle=False))

    def num_tokens_in_example(i):
        return min(self.src_lens[i], self.max_target_length)
    batch_sampler: List[List[int]] = batch_by_size(sorted_indices,
        num_tokens_fn=num_tokens_in_example, max_tokens=
        max_tokens_per_batch, required_batch_size_multiple=64)
    shuffled_batches = [batch_sampler[i] for i in np.random.permutation(
        range(len(batch_sampler)))]
    approximate_toks_per_batch = [(max(self.src_lens[i] for i in batch) *
        len(batch)) for batch in shuffled_batches]
    largest_batch_idx = np.argmax(approximate_toks_per_batch)
    shuffled_batches[0], shuffled_batches[largest_batch_idx
        ] = shuffled_batches[largest_batch_idx], shuffled_batches[0]
    return shuffled_batches
