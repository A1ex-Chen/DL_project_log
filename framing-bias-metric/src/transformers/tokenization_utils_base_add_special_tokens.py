def add_special_tokens(self, special_tokens_dict: Dict[str, Union[str,
    AddedToken]]) ->int:
    """
        Add a dictionary of special tokens (eos, pad, cls, etc.) to the encoder and link them to class attributes. If
        special tokens are NOT in the vocabulary, they are added to it (indexed starting from the last index of the
        current vocabulary).

        Using : obj:`add_special_tokens` will ensure your special tokens can be used in several ways:

        - Special tokens are carefully handled by the tokenizer (they are never split).
        - You can easily refer to special tokens using tokenizer class attributes like :obj:`tokenizer.cls_token`. This
          makes it easy to develop model-agnostic training and fine-tuning scripts.

        When possible, special tokens are already registered for provided pretrained models (for instance
        :class:`~transformers.BertTokenizer` :obj:`cls_token` is already registered to be :obj`'[CLS]'` and XLM's one
        is also registered to be :obj:`'</s>'`).

        Args:
            special_tokens_dict (dictionary `str` to `str` or :obj:`tokenizers.AddedToken`):
                Keys should be in the list of predefined special attributes: [``bos_token``, ``eos_token``,
                ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``, ``mask_token``,
                ``additional_special_tokens``].

                Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer
                assign the index of the ``unk_token`` to them).

        Returns:
            :obj:`int`: Number of tokens added to the vocabulary.

        Examples::

            # Let's see how to add a new classification token to GPT-2
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2Model.from_pretrained('gpt2')

            special_tokens_dict = {'cls_token': '<CLS>'}

            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            print('We have added', num_added_toks, 'tokens')
            # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
            model.resize_token_embeddings(len(tokenizer))

            assert tokenizer.cls_token == '<CLS>'
        """
    if not special_tokens_dict:
        return 0
    added_tokens = 0
    for key, value in special_tokens_dict.items():
        assert key in self.SPECIAL_TOKENS_ATTRIBUTES, f'Key {key} is not a special token'
        if self.verbose:
            logger.info('Assigning %s to the %s key of the tokenizer',
                value, key)
        setattr(self, key, value)
        if key == 'additional_special_tokens':
            assert isinstance(value, (list, tuple)) and all(isinstance(t, (
                str, AddedToken)) for t in value
                ), f'Tokens {value} for key {key} should all be str or AddedToken instances'
            added_tokens += self.add_tokens(value, special_tokens=True)
        else:
            assert isinstance(value, (str, AddedToken)
                ), f'Token {value} for key {key} should be a str or an AddedToken instance'
            added_tokens += self.add_tokens([value], special_tokens=True)
    return added_tokens
