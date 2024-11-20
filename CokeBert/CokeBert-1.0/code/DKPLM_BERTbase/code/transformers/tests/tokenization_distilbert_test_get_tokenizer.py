def get_tokenizer(self, **kwargs):
    return DistilBertTokenizer.from_pretrained(self.tmpdirname, **kwargs)
