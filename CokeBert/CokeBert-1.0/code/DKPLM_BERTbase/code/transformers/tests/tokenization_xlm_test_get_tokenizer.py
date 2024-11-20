def get_tokenizer(self, **kwargs):
    return XLMTokenizer.from_pretrained(self.tmpdirname, **kwargs)
