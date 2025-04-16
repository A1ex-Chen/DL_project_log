def _tokenize(self, text):
    """Returns a tokenized string."""
    return self.sp_model.encode(text, out_type=str)
