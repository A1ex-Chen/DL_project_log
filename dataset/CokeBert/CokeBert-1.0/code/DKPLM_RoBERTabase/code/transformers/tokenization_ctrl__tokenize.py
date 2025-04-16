def _tokenize(self, text):
    """ Tokenize a string.
        """
    split_tokens = []
    text = text.split(' ')
    for token in text:
        split_tokens.extend([t for t in self.bpe(token).split(' ')])
    return split_tokens
