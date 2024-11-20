@property
def vocab_size(self):
    return len(self.sp_model) + self.fairseq_offset + 1
