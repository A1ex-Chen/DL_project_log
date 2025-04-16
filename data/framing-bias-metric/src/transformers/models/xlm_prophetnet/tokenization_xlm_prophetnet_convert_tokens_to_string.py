def convert_tokens_to_string(self, tokens):
    """Converts a sequence of tokens (strings for sub-words) in a single string."""
    out_string = ''.join(tokens).replace(SPIECE_UNDERLINE, ' ').strip()
    return out_string
