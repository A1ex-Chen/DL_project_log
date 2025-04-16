def _maybe_add_prefix_space(self, tokens, decoded):
    if tokens and tokens[0] not in self.no_prefix_space_tokens:
        return ' ' + decoded
    else:
        return decoded
