def _tokenize(self, text):
    """Take as input a string and return a list of strings (tokens) for words/sub-words"""
    if self.do_lower_case:
        text = text.lower()
    return self.gpt2_tokenizer.tokenize(text)
