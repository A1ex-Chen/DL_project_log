def decode(self, tokens):
    text = ''.join([self.decoder[token] for token in tokens])
    text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8',
        errors='replace').replace('</w>', ' ')
    return text
