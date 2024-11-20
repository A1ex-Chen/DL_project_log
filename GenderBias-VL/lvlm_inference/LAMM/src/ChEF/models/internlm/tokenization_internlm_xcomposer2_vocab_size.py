@property
def vocab_size(self):
    """Returns vocab size."""
    return self.sp_model.get_piece_size()
