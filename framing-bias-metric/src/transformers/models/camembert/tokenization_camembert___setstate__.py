def __setstate__(self, d):
    self.__dict__ = d
    self.sp_model = spm.SentencePieceProcessor()
    self.sp_model.Load(self.vocab_file)
