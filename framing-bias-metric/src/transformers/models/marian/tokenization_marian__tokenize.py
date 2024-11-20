def _tokenize(self, text: str) ->List[str]:
    code, text = self.remove_language_code(text)
    pieces = self.current_spm.EncodeAsPieces(text)
    return code + pieces
