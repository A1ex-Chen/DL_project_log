def add_tokens(self, new_tokens):
    """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the
        vocabulary, they are added to it with indices starting from length of the current vocabulary.

        Args:
            new_tokens: list of string. Each string is a token to add. Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer assign the index of the ``unk_token`` to them).

        Returns:
            Number of tokens added to the vocabulary.

        Examples::

            # Let's see how to increase the vocabulary of Bert model and tokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')

            num_added_toks = tokenizer.add_tokens(['new_tok1', 'my_new-tok2'])
            print('We have added', num_added_toks, 'tokens')
            model.resize_token_embeddings(len(tokenizer))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
        """
    if not new_tokens:
        return 0
    to_add_tokens = []
    for token in new_tokens:
        assert isinstance(token, str) or six.PY2 and isinstance(token, unicode)
        if self.init_kwargs.get('do_lower_case', False
            ) and token not in self.all_special_tokens:
            token = token.lower()
        if token != self.unk_token and self.convert_tokens_to_ids(token
            ) == self.convert_tokens_to_ids(self.unk_token
            ) and token not in to_add_tokens:
            to_add_tokens.append(token)
            logger.info('Adding %s to the vocabulary', token)
    added_tok_encoder = dict((tok, len(self) + i) for i, tok in enumerate(
        to_add_tokens))
    added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
    self.added_tokens_encoder.update(added_tok_encoder)
    self.unique_added_tokens_encoder = set(self.added_tokens_encoder.keys()
        ).union(set(self.all_special_tokens))
    self.added_tokens_decoder.update(added_tok_decoder)
    return len(to_add_tokens)
