def tokenize(self, text: TextInput, **kwargs) ->List[str]:
    """
        Converts a string in a sequence of tokens, using the tokenizer.

        Note that, unlike Fast tokenizers (instances of PreTrainedTokenizerFast), this method won't replace the unknown
        tokens with the `unk_token` yet (this is done in the `encode()` method)

        Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies
        (BPE/SentencePieces/WordPieces). Takes care of added tokens.

        Args:
            text (:obj:`str`):
                The sequence to be encoded.
            **kwargs (additional keyword arguments):
                Passed along to the model-specific ``prepare_for_tokenization`` preprocessing method.

        Returns:
            :obj:`List[str]`: The list of tokens.
        """
    all_special_tokens_extended = dict((str(t), t) for t in self.
        all_special_tokens_extended if isinstance(t, AddedToken))
    text, kwargs = self.prepare_for_tokenization(text, **kwargs)
    if kwargs:
        logger.warning(f'Keyword arguments {kwargs} not recognized.')
    if hasattr(self, 'do_lower_case') and self.do_lower_case:
        escaped_special_toks = [re.escape(s_tok) for s_tok in self.
            all_special_tokens]
        pattern = '(' + '|'.join(escaped_special_toks) + ')|' + '(.+?)'
        text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].
            lower(), text)

    def split_on_token(tok, text):
        result = []
        tok_extended = all_special_tokens_extended.get(tok, None)
        split_text = text.split(tok)
        full_word = ''
        for i, sub_text in enumerate(split_text):
            if isinstance(tok_extended, AddedToken):
                if tok_extended.single_word:
                    if i < len(split_text) - 1 and not _is_end_of_word(sub_text
                        ) and not _is_start_of_word(split_text[i + 1]):
                        full_word += sub_text + tok
                    elif full_word:
                        full_word += sub_text
                        result.append(full_word)
                        full_word = ''
                        continue
                if tok_extended.rstrip and i > 0:
                    sub_text = sub_text.lstrip()
                if tok_extended.lstrip and i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
            else:
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()
            if i == 0 and not sub_text:
                result.append(tok)
            elif i == len(split_text) - 1:
                if sub_text:
                    result.append(sub_text)
                else:
                    pass
            else:
                if sub_text:
                    result.append(sub_text)
                result.append(tok)
        return result

    def split_on_tokens(tok_list, text):
        if not text.strip():
            return []
        if not tok_list:
            return self._tokenize(text)
        tokenized_text = []
        text_list = [text]
        for tok in tok_list:
            tokenized_text = []
            for sub_text in text_list:
                if sub_text not in self.unique_no_split_tokens:
                    tokenized_text.extend(split_on_token(tok, sub_text))
                else:
                    tokenized_text.append(sub_text)
            text_list = tokenized_text
        return list(itertools.chain.from_iterable(self._tokenize(token) if 
            token not in self.unique_no_split_tokens else [token] for token in
            tokenized_text))
    no_split_token = self.unique_no_split_tokens
    tokenized_text = split_on_tokens(no_split_token, text)
    return tokenized_text
