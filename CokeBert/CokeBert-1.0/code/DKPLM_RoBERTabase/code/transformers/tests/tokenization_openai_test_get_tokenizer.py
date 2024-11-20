def get_tokenizer(self, **kwargs):
    return OpenAIGPTTokenizer.from_pretrained(self.tmpdirname, **kwargs)
