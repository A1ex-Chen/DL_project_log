def convert_tokens_to_string(self, tokens):
    """ Converts a sequence of tokens (string) in a single string. """
    out_string = self.sp_model.decode_pieces(tokens)
    return out_string
