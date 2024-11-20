def bpe(self, token):
    if token in self.cache:
        return self.cache[token]
    word = tuple(token[:-1]) + (token[-1] + '</w>',)
    pairs = get_pairs(word)
    if not pairs:
        return token + '</w>'
    while True:
        bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float
            ('inf')))
        if bigram not in self.bpe_ranks:
            break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j
            except:
                new_word.extend(word[i:])
                break
            if word[i] == first and i < len(word) - 1 and word[i + 1
                ] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)
        word = new_word
        if len(word) == 1:
            break
        else:
            pairs = get_pairs(word)
    word = ' '.join(word)
    self.cache[token] = word
    return word
