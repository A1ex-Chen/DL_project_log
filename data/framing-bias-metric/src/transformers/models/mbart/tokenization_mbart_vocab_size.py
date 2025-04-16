@property
def vocab_size(self):
    return len(self.sp_model) + len(self.lang_code_to_id
        ) + self.fairseq_offset + 1
