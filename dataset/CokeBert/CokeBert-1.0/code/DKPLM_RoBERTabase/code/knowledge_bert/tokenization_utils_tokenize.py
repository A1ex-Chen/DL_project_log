def tokenize(self, text, **kwargs):
    """ Converts a string in a sequence of tokens (string), using the tokenizer.
            Split in words for word-based vocabulary or sub-words for sub-word-based
            vocabularies (BPE/SentencePieces/WordPieces).

            Take care of added tokens.

            text: The sequence to be encoded.
            **kwargs: passed to the child `self.tokenize()` method
        """
    all_special_tokens = self.all_special_tokens

    def lowercase_text(t):
        escaped_special_toks = [re.escape(s_tok) for s_tok in
            all_special_tokens]
        pattern = '(' + '|'.join(escaped_special_toks) + ')|' + '(.+?)'
        return re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].
            lower(), t)
    if self.init_kwargs.get('do_lower_case', False):
        text = lowercase_text(text)

    def split_on_token(tok, text):
        result = []
        split_text = text.split(tok)
        for i, sub_text in enumerate(split_text):
            sub_text = sub_text.strip()
            if i == 0 and not sub_text:
                result += [tok]
            elif i == len(split_text) - 1:
                if sub_text:
                    result += [sub_text]
                else:
                    pass
            else:
                if sub_text:
                    result += [sub_text]
                result += [tok]
        return result

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
    added_tokens = self.unique_added_tokens_encoder
    tokenized_text = split_on_tokens(added_tokens, text)
    return tokenized_text
