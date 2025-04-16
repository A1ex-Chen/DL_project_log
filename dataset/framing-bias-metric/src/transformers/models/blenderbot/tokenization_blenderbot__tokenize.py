def _tokenize(self, text: str) ->List[str]:
    """ Split a string into tokens using BPE."""
    split_tokens = []
    words = re.findall('\\S+\\n?', text)
    for token in words:
        split_tokens.extend([t for t in self.bpe(token).split(' ')])
    return split_tokens
