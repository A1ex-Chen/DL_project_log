def _convert_token_to_id(self, token):
    """ Converts a token (str) in an id using the vocab. """
    if token.startswith('<extra_id_'):
        match = re.match('<extra_id_(\\d+)>', token)
        num = int(match.group(1))
        return self.vocab_size - num - 1
    return self.sp_model.piece_to_id(token)
