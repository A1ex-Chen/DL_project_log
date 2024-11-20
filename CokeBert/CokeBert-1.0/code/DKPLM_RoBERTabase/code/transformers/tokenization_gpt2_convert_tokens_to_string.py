def convert_tokens_to_string(self, tokens):
    """ Converts a sequence of tokens (string) in a single string. """
    text = ''.join(tokens)
    text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8',
        errors=self.errors)
    return text
