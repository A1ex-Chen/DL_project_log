def add_word(self, word):
    if word not in self.word2idx:
        self.idx2word.append(word)
        self.word2idx[word] = len(self.idx2word) - 1
    return self.word2idx[word]
