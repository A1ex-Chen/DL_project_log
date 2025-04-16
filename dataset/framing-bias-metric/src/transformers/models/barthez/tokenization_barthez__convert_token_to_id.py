def _convert_token_to_id(self, token):
    """ Converts a token (str) in an id using the vocab. """
    if token in self.fairseq_tokens_to_ids:
        return self.fairseq_tokens_to_ids[token]
    spm_id = self.sp_model.PieceToId(token)
    return spm_id if spm_id else self.unk_token_id
