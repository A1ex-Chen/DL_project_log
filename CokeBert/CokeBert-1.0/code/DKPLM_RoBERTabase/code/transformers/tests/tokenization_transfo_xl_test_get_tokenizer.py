def get_tokenizer(self, **kwargs):
    kwargs['lower_case'] = True
    return TransfoXLTokenizer.from_pretrained(self.tmpdirname, **kwargs)
