@property
def vocab_size(self):
    return len(self.fairseq_tokens_to_ids) + len(self.sp_model)
