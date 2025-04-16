def calc_banned_bad_words_ids(prev_input_ids, bad_words_ids):
    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            return True
        if len(tokens) > len(prev_tokens):
            return False
        if prev_tokens[-len(tokens):] == tokens:
            return True
        else:
            return False
    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []
        for banned_token_seq in bad_words_ids:
            assert len(banned_token_seq
                ) > 0, 'Banned words token sequences {} cannot have an empty list'.format(
                bad_words_ids)
            if _tokens_match(prev_input_ids_slice.numpy().tolist(),
                banned_token_seq[:-1]) is False:
                continue
            banned_tokens_slice.append(banned_token_seq[-1])
        banned_tokens.append(banned_tokens_slice)
    return banned_tokens
