def _tokenize(self, text):
    """ Tokenize a string. """
    bpe_tokens = []
    for token in re.findall(self.pat, text):
        token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
        bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(' ')
            )
    return bpe_tokens
