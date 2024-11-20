def _convert_id_to_token(self, index: int) ->str:
    """Converts an index (integer) to a token (str) using the vocab."""
    if index in self.encoder:
        return self.encoder[index]
    elif index in self.added_tokens_encoder:
        return self.added_tokens_encoder[index]
    else:
        token = self.sp_model.IdToPiece(index - self.offset)
    return token
