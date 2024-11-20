def get_tokenizer(self, **kwargs):
    kwargs.update(self.special_tokens_map)
    return CTRLTokenizer.from_pretrained(self.tmpdirname, **kwargs)
