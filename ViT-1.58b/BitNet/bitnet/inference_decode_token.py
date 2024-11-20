@staticmethod
def decode_token(token):
    """Decodes a token into a character."""
    return str(chr(max(32, token)))
