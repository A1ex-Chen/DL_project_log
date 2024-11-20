def _tokenize(self, text):
    split_tokens = []
    if self.do_basic_tokenize:
        for token in self.basic_tokenizer.tokenize(text, never_split=self.
            all_special_tokens):
            if token in self.basic_tokenizer.never_split:
                split_tokens.append(token)
            else:
                split_tokens += self.wordpiece_tokenizer.tokenize(token)
    else:
        split_tokens = self.wordpiece_tokenizer.tokenize(text)
    return split_tokens
