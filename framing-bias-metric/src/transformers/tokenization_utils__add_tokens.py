def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]],
    special_tokens: bool=False) ->int:
    """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
        it with indices starting from length of the current vocabulary.

        Args:
            new_tokens (:obj:`List[str]`or :obj:`List[tokenizers.AddedToken]`):
                Token(s) to add in vocabulary. A token is only added if it's not already in the vocabulary (tested by
                checking if the tokenizer assign the index of the ``unk_token`` to them).
            special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the tokens should be added as special tokens.

        Returns:
            :obj:`int`: The number of tokens actually added to the vocabulary.

        Examples::

            # Let's see how to increase the vocabulary of Bert model and tokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')

            num_added_toks = tokenizer.add_tokens(['new_tok1', 'my_new-tok2'])
            print('We have added', num_added_toks, 'tokens')
            # Note: resize_token_embeddings expects to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
            model.resize_token_embeddings(len(tokenizer))
        """
    new_tokens = [str(tok) for tok in new_tokens]
    tokens_to_add = []
    for token in new_tokens:
        assert isinstance(token, str)
        if not special_tokens and hasattr(self, 'do_lower_case'
            ) and self.do_lower_case:
            token = token.lower()
        if token != self.unk_token and self.convert_tokens_to_ids(token
            ) == self.convert_tokens_to_ids(self.unk_token
            ) and token not in tokens_to_add:
            tokens_to_add.append(token)
            if self.verbose:
                logger.info('Adding %s to the vocabulary', token)
    added_tok_encoder = dict((tok, len(self) + i) for i, tok in enumerate(
        tokens_to_add))
    added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
    self.added_tokens_encoder.update(added_tok_encoder)
    self.added_tokens_decoder.update(added_tok_decoder)
    if special_tokens:
        self.unique_no_split_tokens = sorted(set(self.
            unique_no_split_tokens).union(set(new_tokens)))
    else:
        self.unique_no_split_tokens = sorted(set(self.
            unique_no_split_tokens).union(set(tokens_to_add)))
    return len(tokens_to_add)
