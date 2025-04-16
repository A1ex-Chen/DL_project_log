def _tokenize(self, text):
    pre_tokens = self.bert_pre_tokenizer.tokenize(text)
    split_tokens = []
    for token in pre_tokens:
        if token:
            split_tokens.extend([t for t in self.bpe(token).split(' ')])
    return split_tokens
