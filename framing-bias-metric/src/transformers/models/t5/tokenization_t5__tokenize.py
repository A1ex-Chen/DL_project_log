def _tokenize(self, text, sample=False):
    """Take as input a string and return a list of strings (tokens) for words/sub-words"""
    if not sample:
        pieces = self.sp_model.EncodeAsPieces(text)
    else:
        pieces = self.sp_model.SampleEncodeAsPieces(text, 64, 0.1)
    return pieces
