def _tokenize(self, text):
    split_tokens = []
    if self.do_basic_tokenize:
        for token in self.basic_tokenizer.tokenize(text, never_split=self.
            all_special_tokens):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
    else:
        split_tokens = self.wordpiece_tokenizer.tokenize(text)
    return split_tokens
