def convert_tokens_to_string(self, tokens):
    out_string = ''.join(tokens).replace(SPIECE_UNDERLINE, ' ').strip()
    return out_string
