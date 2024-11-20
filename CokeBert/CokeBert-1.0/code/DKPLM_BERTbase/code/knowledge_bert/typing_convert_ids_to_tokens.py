def convert_ids_to_tokens(self, ids):
    """Converts a sequence of ids in wordpiece tokens using the vocab."""
    tokens = []
    for i in ids:
        tokens.append(self.ids_to_tokens[i])
    return tokens
