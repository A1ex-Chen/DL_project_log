def _convert_id_to_token(self, index):
    """Converts an index (integer) in a token (str) using the vocab."""
    if index in self.fairseq_ids_to_tokens:
        return self.fairseq_ids_to_tokens[index]
    return self.sp_model.IdToPiece(index - self.fairseq_offset)
