def get_tokenizer(self, **kwargs):
    kwargs.update(self.special_tokens_map)
    return GPT2Tokenizer.from_pretrained(self.tmpdirname, **kwargs)
