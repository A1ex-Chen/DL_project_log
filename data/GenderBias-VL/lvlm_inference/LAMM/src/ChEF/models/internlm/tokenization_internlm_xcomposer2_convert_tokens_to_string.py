def convert_tokens_to_string(self, tokens):
    """Converts a sequence of tokens (string) in a single string."""
    current_sub_tokens = []
    out_string = ''
    prev_is_special = False
    for token in tokens:
        if token in self.all_special_tokens:
            if not prev_is_special:
                out_string += ' '
            out_string += self.sp_model.decode(current_sub_tokens) + token
            prev_is_special = True
            current_sub_tokens = []
        else:
            current_sub_tokens.append(token)
            prev_is_special = False
    out_string += self.sp_model.decode(current_sub_tokens)
    out_string = self.clean_up_tokenization(out_string)
    out_string = self._maybe_add_prefix_space(tokens=tokens, decoded=out_string
        )
    return out_string[1:]
