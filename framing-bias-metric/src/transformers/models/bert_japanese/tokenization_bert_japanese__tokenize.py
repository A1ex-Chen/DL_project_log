def _tokenize(self, text):
    if self.do_word_tokenize:
        tokens = self.word_tokenizer.tokenize(text, never_split=self.
            all_special_tokens)
    else:
        tokens = [text]
    if self.do_subword_tokenize:
        split_tokens = [sub_token for token in tokens for sub_token in self
            .subword_tokenizer.tokenize(token)]
    else:
        split_tokens = tokens
    return split_tokens
