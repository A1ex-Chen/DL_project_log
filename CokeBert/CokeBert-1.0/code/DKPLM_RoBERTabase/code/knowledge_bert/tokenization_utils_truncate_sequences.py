def truncate_sequences(self, ids, pair_ids=None, num_tokens_to_remove=0,
    truncation_strategy='longest_first', stride=0):
    """Truncates a sequence pair in place to the maximum length.
            truncation_strategy: string selected in the following options:
                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
                    starting from the longest one at each token (when there is a pair of input sequences).
                    Overflowing tokens only contains overflow from the first sequence.
                - 'only_first': Only truncate the first sequence. raise an error if the first sequence is shorter or equal to than num_tokens_to_remove.
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_length)
        """
    if num_tokens_to_remove <= 0:
        return ids, pair_ids, []
    if truncation_strategy == 'longest_first':
        overflowing_tokens = []
        for _ in range(num_tokens_to_remove):
            if pair_ids is None or len(ids) > len(pair_ids):
                overflowing_tokens = [ids[-1]] + overflowing_tokens
                ids = ids[:-1]
            else:
                pair_ids = pair_ids[:-1]
        window_len = min(len(ids), stride)
        if window_len > 0:
            overflowing_tokens = ids[-window_len:] + overflowing_tokens
    elif truncation_strategy == 'only_first':
        assert len(ids) > num_tokens_to_remove
        window_len = min(len(ids), stride + num_tokens_to_remove)
        overflowing_tokens = ids[-window_len:]
        ids = ids[:-num_tokens_to_remove]
    elif truncation_strategy == 'only_second':
        assert pair_ids is not None and len(pair_ids) > num_tokens_to_remove
        window_len = min(len(pair_ids), stride + num_tokens_to_remove)
        overflowing_tokens = pair_ids[-window_len:]
        pair_ids = pair_ids[:-num_tokens_to_remove]
    elif truncation_strategy == 'do_not_truncate':
        raise ValueError(
            'Input sequence are too long for max_length. Please select a truncation strategy.'
            )
    else:
        raise ValueError(
            "Truncation_strategy should be selected in ['longest_first', 'only_first', 'only_second', 'do_not_truncate']"
            )
    return ids, pair_ids, overflowing_tokens
