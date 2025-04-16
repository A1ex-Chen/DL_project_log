@property
def vocab_size(self):
    return self.fairseq_offset + len(self.sp_model)
