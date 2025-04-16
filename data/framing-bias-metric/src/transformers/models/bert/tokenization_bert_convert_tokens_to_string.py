def convert_tokens_to_string(self, tokens):
    """ Converts a sequence of tokens (string) in a single string. """
    out_string = ' '.join(tokens).replace(' ##', '').strip()
    return out_string
