def _tokenize(self, text):
    """Tokenize a string."""
    split_tokens = []
    words = re.findall('\\S+\\n?', text)
    for token in words:
        split_tokens.extend([t for t in self.bpe(token).split(' ')])
    return split_tokens
