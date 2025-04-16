def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(' ')
    if isinstance(word_place, str):
        word_place = [i for i, word in enumerate(split_text) if word_place ==
            word]
    elif isinstance(word_place, int):
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip('#') for item in
            tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0
        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)
