def tokenize(self, text, **kwargs):
    """ Converts a string in a sequence of tokens (string), using the tokenizer.
            Split in words for word-based vocabulary or sub-words for sub-word-based
            vocabularies (BPE/SentencePieces/WordPieces).

            Take care of added tokens.
        """

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
        if not text:
            return []
        if not tok_list:
            return self._tokenize(text, **kwargs)
        tokenized_text = []
        text_list = [text]
        for tok in tok_list:
            tokenized_text = []
            for sub_text in text_list:
                if (sub_text not in self.added_tokens_encoder and sub_text
                     not in self.all_special_tokens):
                    tokenized_text += split_on_token(tok, sub_text)
                else:
                    tokenized_text += [sub_text]
            text_list = tokenized_text
        return list(itertools.chain.from_iterable(self._tokenize(token, **
            kwargs) if token not in self.added_tokens_encoder and token not in
            self.all_special_tokens and token not in ['[unused1]',
            '[unused2]', '[unused3]', '[unused4]'] else [token] for token in
            tokenized_text))
    added_tokens = list(self.added_tokens_encoder.keys()
        ) + self.all_special_tokens + ['[unused1]', '[unused2]',
        '[unused3]', '[unused4]']
    tokenized_text = split_on_tokens(added_tokens, text)
    return tokenized_text
