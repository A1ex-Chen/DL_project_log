def _convert_token_to_id(self, token):
    """ Converts a token (str/unicode) in an id using the vocab. """
    if token in self.fairseq_tokens_to_ids:
        return self.fairseq_tokens_to_ids[token]
    return self.fairseq_offset + self.sp_model.PieceToId(token)