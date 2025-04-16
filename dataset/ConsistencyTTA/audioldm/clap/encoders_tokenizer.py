def tokenizer(self, text):
    result = self.tokenize(text, padding='max_length', truncation=True,
        max_length=512, return_tensors='pt')
    return {k: v.squeeze(0) for k, v in result.items()}
