def _convert_id_to_token(self, index):
    """Converts an index (integer) in a token (str) using the vocab."""
    if index < self.sp_model.get_piece_size():
        token = self.sp_model.IdToPiece(index)
    return token
