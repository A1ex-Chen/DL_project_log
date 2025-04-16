def convert_tokens_to_string(self, tokens):
    """ Converts a sequence of tokens (string) in a single string. """
    current_sub_tokens = []
    out_string = ''
    for token in tokens:
        if token in self.all_special_tokens:
            out_string += self.sp_model.decode_pieces(current_sub_tokens
                ) + token + ' '
            current_sub_tokens = []
        else:
            current_sub_tokens.append(token)
    out_string += self.sp_model.decode_pieces(current_sub_tokens)
    return out_string.strip()
