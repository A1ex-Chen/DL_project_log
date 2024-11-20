@staticmethod
def decode_tokens(tokens):
    """Decodes a sequence of tokens into a string."""
    return ''.join(list(map(BitNetInference.decode_token, tokens)))
