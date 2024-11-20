@property
def vocab_size(self) ->int:
    return len(self.sp_model) + self.offset
