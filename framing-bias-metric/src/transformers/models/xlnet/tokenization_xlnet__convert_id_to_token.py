def _convert_id_to_token(self, index):
    """Converts an index (integer) in a token (str) using the vocab."""
    return self.sp_model.IdToPiece(index)
