def get_tokens(self, text):
    return self.model._tokenizer.tokenize(text)
