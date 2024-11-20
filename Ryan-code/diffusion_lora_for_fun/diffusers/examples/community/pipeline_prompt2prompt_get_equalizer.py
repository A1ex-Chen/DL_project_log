def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]],
    values: Union[List[float], Tuple[float, ...]], tokenizer):
    if isinstance(word_select, (int, str)):
        word_select = word_select,
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer
