def split_on_tokens(tok_list, text):
    if not text:
        return []
    if not tok_list:
        return self._tokenize(text, **kwargs)
    tokenized_text = []
    text_list = [text]
    for tok in tok_list:
        tokenized_text = []
        for sub_text in text_list:
            if (sub_text not in self.added_tokens_encoder and sub_text not in
                self.all_special_tokens):
                tokenized_text += split_on_token(tok, sub_text)
            else:
                tokenized_text += [sub_text]
        text_list = tokenized_text
    return list(itertools.chain.from_iterable(self._tokenize(token, **
        kwargs) if token not in self.added_tokens_encoder and token not in
        self.all_special_tokens and token not in ['[unused1]', '[unused2]',
        '[unused3]', '[unused4]'] else [token] for token in tokenized_text))
