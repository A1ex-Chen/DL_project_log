def split_on_tokens(tok_list, text):
    if not text.strip():
        return []
    if not tok_list:
        return self._tokenize(text, **kwargs)
    tokenized_text = []
    text_list = [text]
    for tok in tok_list:
        tokenized_text = []
        for sub_text in text_list:
            if sub_text not in self.unique_added_tokens_encoder:
                tokenized_text += split_on_token(tok, sub_text)
            else:
                tokenized_text += [sub_text]
        text_list = tokenized_text
    return list(itertools.chain.from_iterable(self._tokenize(token, **
        kwargs) if token not in self.unique_added_tokens_encoder else [
        token] for token in tokenized_text))
