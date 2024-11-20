def add_symbol(self, word, n=1):
    """
        Adds a word to the dictionary

        Args:
          word (:obj:`str`): Tthe new token/word to be added to the vocabulary.
          n (int, optional): The frequency of the word.

        Returns:
          The id of the new word.

        """
    if word in self.indices:
        idx = self.indices[word]
        self.count[idx] = self.count[idx] + n
        return idx
    else:
        idx = len(self.symbols)
        self.indices[word] = idx
        self.symbols.append(word)
        self.count.append(n)
        return idx
