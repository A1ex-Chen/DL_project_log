def get_tokenizer(self, **kwargs):
    return BertTokenizer.from_pretrained(self.tmpdirname, **kwargs)
