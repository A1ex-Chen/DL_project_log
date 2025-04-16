def non_pad_len(tokens: np.ndarray) ->int:
    return np.count_nonzero(tokens != tokenizer.pad_token_id)
