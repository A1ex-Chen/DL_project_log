def encode_plus(self, text, text_pair=None, add_special_tokens=True,
    max_length=None, stride=0, truncation_strategy='longest_first',
    return_tensors=None, **kwargs):
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
            return_tensors: (optional) can be set to 'tf' or 'pt' to return respectively TensorFlow tf.constant
                or PyTorch torch.Tensor instead of a list of python integers.
            **kwargs: passed to the `self.tokenize()` method
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
        max_length=max_length, add_special_tokens=add_special_tokens,
        stride=stride, truncation_strategy=truncation_strategy,
        return_tensors=return_tensors)
