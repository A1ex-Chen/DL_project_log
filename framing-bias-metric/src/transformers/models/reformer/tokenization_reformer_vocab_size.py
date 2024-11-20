@property
def vocab_size(self):
    return self.sp_model.get_piece_size()
