def _tokenize(self, text):
    """ Tokenize a string. """
    split_tokens = []
    if self.fix_text is None:
        text = self.nlp.tokenize(text)
        for token in text:
            split_tokens.extend([t for t in self.bpe(token).split(' ')])
    else:
        text = self.nlp(text_standardize(self.fix_text(text)))
        for token in text:
            split_tokens.extend([t for t in self.bpe(token.text.lower()).
                split(' ')])
    return split_tokens
