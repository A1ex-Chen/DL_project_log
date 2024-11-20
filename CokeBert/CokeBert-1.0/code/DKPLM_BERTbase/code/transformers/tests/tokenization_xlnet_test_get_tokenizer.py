def get_tokenizer(self, **kwargs):
    return XLNetTokenizer.from_pretrained(self.tmpdirname, **kwargs)
