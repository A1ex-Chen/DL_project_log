def encode_plus(self, text, text_pair=None, add_special_tokens=True,
    max_length=None, stride=0, truncation_strategy='longest_first',
    pad_to_max_length=False, return_tensors=None, return_token_type_ids=
    True, return_attention_mask=True, return_overflowing_tokens=False,
    return_special_tokens_mask=False, **kwargs):
    """
        Returns a dictionary containing the encoded sequence or sequence pair and additional informations:
        the mask for sequence classification and the overflowing elements if a ``max_length`` is specified.

        Args:
            text: The first sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method)
            text_pair: Optional second sequence to be encoded. This can be a string, a list of strings (tokenized
                string using the `tokenize` method) or a list of integers (tokenized string ids using the
                `convert_tokens_to_ids` method)
            add_special_tokens: if set to ``True``, the sequences will be encoded with the special tokens relative
                to their model.
            max_length: if set to a number, will limit the total sequence returned so that it has a maximum length.
                If there are overflowing tokens, those will be added to the returned dictionary
            stride: if set to a number along with max_length, the overflowing tokens returned will contain some tokens
                from the main sequence returned. The value of this argument defines the number of additional tokens.
            truncation_strategy: string selected in the following options:
                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
                    starting from the longest one at each token (when there is a pair of input sequences)
                - 'only_first': Only truncate the first sequence
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_length)
            pad_to_max_length: if set to True, the returned sequences will be padded according to the model's padding side and
                padding index, up to their max length. If no max length is specified, the padding is done up to the model's max length.
                The tokenizer padding sides are handled by the following strings:
                - 'left': pads on the left of the sequences
                - 'right': pads on the right of the sequences
                Defaults to False: no padding.
            return_tensors: (optional) can be set to 'tf' or 'pt' to return respectively TensorFlow tf.constant
                or PyTorch torch.Tensor instead of a list of python integers.
            return_token_type_ids: (optional) Set to False to avoid returning token_type_ids (default True).
            return_attention_mask: (optional) Set to False to avoir returning attention mask (default True)
            return_overflowing_tokens: (optional) Set to True to return overflowing token information (default False).
            return_special_tokens_mask: (optional) Set to True to return special tokens mask information (default False).
            **kwargs: passed to the `self.tokenize()` method

        Return:
            A Dictionary of shape::

                {
                    input_ids: list[int],
                    token_type_ids: list[int] if return_token_type_ids is True (default)
                    attention_mask: list[int] if return_attention_mask is True (default)
                    overflowing_tokens: list[int] if a ``max_length`` is specified and return_overflowing_tokens is True
                    num_truncated_tokens: int if a ``max_length`` is specified and return_overflowing_tokens is True
                    special_tokens_mask: list[int] if ``add_special_tokens`` if set to ``True`` and return_special_tokens_mask is True
                }

            With the fields:
                ``input_ids``: list of token ids to be fed to a model
                ``token_type_ids``: list of token type ids to be fed to a model
                ``attention_mask``: list of indices specifying which tokens should be attended to by the model
                ``overflowing_tokens``: list of overflowing tokens if a max length is specified.
                ``num_truncated_tokens``: number of overflowing tokens a ``max_length`` is specified
                ``special_tokens_mask``: if adding special tokens, this is a list of [0, 1], with 0 specifying special added
                tokens and 1 specifying sequence tokens.
        """

    def get_input_ids(text):
        if isinstance(text, six.string_types):
            return self.convert_tokens_to_ids(self.tokenize(text, **kwargs))
        elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(
            text[0], six.string_types):
            return self.convert_tokens_to_ids(text)
        elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(
            text[0], int):
            return text
        else:
            raise ValueError(
                'Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers.'
                )
    first_ids = get_input_ids(text)
    second_ids = get_input_ids(text_pair) if text_pair is not None else None
    return self.prepare_for_model(first_ids, pair_ids=second_ids,
        max_length=max_length, pad_to_max_length=pad_to_max_length,
        add_special_tokens=add_special_tokens, stride=stride,
        truncation_strategy=truncation_strategy, return_tensors=
        return_tensors, return_attention_mask=return_attention_mask,
        return_token_type_ids=return_token_type_ids,
        return_overflowing_tokens=return_overflowing_tokens,
        return_special_tokens_mask=return_special_tokens_mask)
