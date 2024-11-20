def encode(self, text):
    bpe_tokens = []
    text = whitespace_clean(basic_clean(text)).lower()
    for token in re.findall(self.pat, text):
        token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
        bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe
            (token).split(' '))
    return bpe_tokens
