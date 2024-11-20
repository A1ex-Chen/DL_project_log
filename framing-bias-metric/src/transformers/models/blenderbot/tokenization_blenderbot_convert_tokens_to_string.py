def convert_tokens_to_string(self, tokens: List[str]) ->str:
    """ Converts a sequence of tokens  in a single string. """
    out_string = ' '.join(tokens).replace('@@ ', '').strip()
    return out_string
