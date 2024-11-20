def _tokenize(self, text):
    return self.sp_model.EncodeAsPieces(text)
