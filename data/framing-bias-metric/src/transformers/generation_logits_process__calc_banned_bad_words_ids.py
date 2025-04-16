def _calc_banned_bad_words_ids(self, prev_input_ids: Iterable[int]) ->Iterable[
    int]:
    banned_tokens = []
    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []
        for banned_token_seq in self.bad_words_ids:
            if self._tokens_match(prev_input_ids_slice, banned_token_seq[:-1]
                ) is False:
                continue
            banned_tokens_slice.append(banned_token_seq[-1])
        banned_tokens.append(banned_tokens_slice)
    return banned_tokens
