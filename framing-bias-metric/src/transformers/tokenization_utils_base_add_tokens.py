def add_tokens(self, new_tokens: Union[str, AddedToken, List[Union[str,
    AddedToken]]], special_tokens: bool=False) ->int:
    """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
        it with indices starting from length of the current vocabulary.

        Args:
            new_tokens (:obj:`str`, :obj:`tokenizers.AddedToken` or a list of `str` or :obj:`tokenizers.AddedToken`):
                Tokens are only added if they are not already in the vocabulary. :obj:`tokenizers.AddedToken` wraps a
                string token to let you personalize its behavior: whether this token should only match against a single
                word, whether this token should strip all potential whitespaces on the left side, whether this token
                should strip all potential whitespaces on the right side, etc.
            special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Can be used to specify if the token is a special token. This mostly change the normalization behavior
                (special tokens like CLS or [MASK] are usually not lower-cased for instance).

                See details for :obj:`tokenizers.AddedToken` in HuggingFace tokenizers library.

        Returns:
            :obj:`int`: Number of tokens added to the vocabulary.

        Examples::

            # Let's see how to increase the vocabulary of Bert model and tokenizer
            tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')

            num_added_toks = tokenizer.add_tokens(['new_tok1', 'my_new-tok2'])
            print('We have added', num_added_toks, 'tokens')
             # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
            model.resize_token_embeddings(len(tokenizer))
        """
    if not new_tokens:
        return 0
    if not isinstance(new_tokens, (list, tuple)):
        new_tokens = [new_tokens]
    return self._add_tokens(new_tokens, special_tokens=special_tokens)
