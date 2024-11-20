def __init__(self, stop_words_ids: Iterable[Iterable[int]], eos_token_id: int):
    if not isinstance(stop_words_ids, List) or len(stop_words_ids) == 0:
        raise ValueError(
            f'`stop_words_ids` has to be a non-emtpy list, but is {stop_words_ids}.'
            )
    if any(not isinstance(bad_word_ids, list) for bad_word_ids in
        stop_words_ids):
        raise ValueError(
            f'`stop_words_ids` has to be a list of lists, but is {stop_words_ids}.'
            )
    if any(any(not isinstance(token_id, (int, np.integer)) or token_id < 0 for
        token_id in stop_word_ids) for stop_word_ids in stop_words_ids):
        raise ValueError(
            f'Each list in `stop_words_ids` has to be a list of positive integers, but is {stop_words_ids}.'
            )
    self.stop_words_ids = list(filter(lambda bad_token_seq: bad_token_seq !=
        [eos_token_id], stop_words_ids))
    self.eos_token_id = eos_token_id
    for stop_token_seq in self.stop_words_ids:
        assert len(stop_token_seq
            ) > 0, 'Stop words token sequences {} cannot have an empty list'.format(
            stop_words_ids)
