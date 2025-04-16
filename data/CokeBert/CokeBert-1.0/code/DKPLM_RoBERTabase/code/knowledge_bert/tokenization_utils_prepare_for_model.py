def prepare_for_model(self, ids, pair_ids=None, max_length=None,
    add_special_tokens=True, stride=0, truncation_strategy='longest_first',
    pad_to_max_length=False, return_tensors=None, return_token_type_ids=
    True, return_attention_mask=True, return_overflowing_tokens=False,
    return_special_tokens_mask=False):
    """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model.
        It adds special tokens, truncates
        sequences if overflowing while taking into account the special tokens and manages a window stride for
        overflowing tokens

        Args:
            ids: list of tokenized input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            pair_ids: Optional second list of input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            max_length: maximum length of the returned list. Will truncate by taking into account the special tokens.
            add_special_tokens: if set to ``True``, the sequences will be encoded with the special tokens relative
                to their model.
            stride: window stride for overflowing tokens. Can be useful for edge effect removal when using sequential
                list of inputs.
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
            return_attention_mask: (optional) Set to False to avoid returning attention mask (default True)
            return_overflowing_tokens: (optional) Set to True to return overflowing token information (default False).
            return_special_tokens_mask: (optional) Set to True to return special tokens mask information (default False).

        Return:
            A Dictionary of shape::

                {
                    input_ids: list[int],
                    token_type_ids: list[int] if return_token_type_ids is True (default)
                    overflowing_tokens: list[int] if a ``max_length`` is specified and return_overflowing_tokens is True
                    num_truncated_tokens: int if a ``max_length`` is specified and return_overflowing_tokens is True
                    special_tokens_mask: list[int] if ``add_special_tokens`` if set to ``True`` and return_special_tokens_mask is True
                }

            With the fields:
                ``input_ids``: list of token ids to be fed to a model
                ``token_type_ids``: list of token type ids to be fed to a model

                ``overflowing_tokens``: list of overflowing tokens if a max length is specified.
                ``num_truncated_tokens``: number of overflowing tokens a ``max_length`` is specified
                ``special_tokens_mask``: if adding special tokens, this is a list of [0, 1], with 0 specifying special added
                tokens and 1 specifying sequence tokens.
        """
    pair = bool(pair_ids is not None)
    len_ids = len(ids)
    len_pair_ids = len(pair_ids) if pair else 0
    encoded_inputs = {}
    total_len = len_ids + len_pair_ids + (self.num_added_tokens(pair=pair) if
        add_special_tokens else 0)
    if max_length and total_len > max_length:
        ids, pair_ids, overflowing_tokens = self.truncate_sequences(ids,
            pair_ids=pair_ids, num_tokens_to_remove=total_len - max_length,
            truncation_strategy=truncation_strategy, stride=stride)
        if return_overflowing_tokens:
            encoded_inputs['overflowing_tokens'] = overflowing_tokens
            encoded_inputs['num_truncated_tokens'] = total_len - max_length
    if add_special_tokens:
        sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
        token_type_ids = self.create_token_type_ids_from_sequences(ids,
            pair_ids)
    else:
        sequence = ids + pair_ids if pair else ids
        token_type_ids = [0] * len(ids) + ([1] * len(pair_ids) if pair else [])
    if return_special_tokens_mask:
        encoded_inputs['special_tokens_mask'] = self.get_special_tokens_mask(
            ids, pair_ids)
    encoded_inputs['input_ids'] = sequence
    if return_token_type_ids:
        encoded_inputs['token_type_ids'] = token_type_ids
    if max_length and len(encoded_inputs['input_ids']) > max_length:
        encoded_inputs['input_ids'] = encoded_inputs['input_ids'][:max_length]
        if return_token_type_ids:
            encoded_inputs['token_type_ids'] = encoded_inputs['token_type_ids'
                ][:max_length]
        if return_special_tokens_mask:
            encoded_inputs['special_tokens_mask'] = encoded_inputs[
                'special_tokens_mask'][:max_length]
    if max_length is None and len(encoded_inputs['input_ids']) > self.max_len:
        logger.warning(
            'Token indices sequence length is longer than the specified maximum sequence length for this model ({} > {}). Running this sequence through the model will result in indexing errors'
            .format(len(ids), self.max_len))
    needs_to_be_padded = pad_to_max_length and (max_length and len(
        encoded_inputs['input_ids']) < max_length or max_length is None and
        len(encoded_inputs['input_ids']) < self.max_len and self.max_len <=
        10000)
    if pad_to_max_length and max_length is None and self.max_len > 10000:
        logger.warning(
            "Sequence can't be padded as no maximum length is specified and the model maximum length is too high."
            )
    if needs_to_be_padded:
        difference = (max_length if max_length is not None else self.max_len
            ) - len(encoded_inputs['input_ids'])
        if self.padding_side == 'right':
            if return_attention_mask:
                encoded_inputs['attention_mask'] = [1] * len(encoded_inputs
                    ['input_ids']) + [0] * difference
            if return_token_type_ids:
                encoded_inputs['token_type_ids'] = encoded_inputs[
                    'token_type_ids'] + [self.pad_token_type_id] * difference
            if return_special_tokens_mask:
                encoded_inputs['special_tokens_mask'] = encoded_inputs[
                    'special_tokens_mask'] + [1] * difference
            encoded_inputs['input_ids'] = encoded_inputs['input_ids'] + [self
                .pad_token_id] * difference
        elif self.padding_side == 'left':
            if return_attention_mask:
                encoded_inputs['attention_mask'] = [0] * difference + [1
                    ] * len(encoded_inputs['input_ids'])
            if return_token_type_ids:
                encoded_inputs['token_type_ids'] = [self.pad_token_type_id
                    ] * difference + encoded_inputs['token_type_ids']
            if return_special_tokens_mask:
                encoded_inputs['special_tokens_mask'] = [1
                    ] * difference + encoded_inputs['special_tokens_mask']
            encoded_inputs['input_ids'] = [self.pad_token_id
                ] * difference + encoded_inputs['input_ids']
        else:
            raise ValueError('Invalid padding strategy:' + str(self.
                padding_side))
    elif return_attention_mask:
        encoded_inputs['attention_mask'] = [1] * len(encoded_inputs[
            'input_ids'])
    if return_tensors == 'tf' and is_tf_available():
        encoded_inputs['input_ids'] = tf.constant([encoded_inputs['input_ids']]
            )
        encoded_inputs['token_type_ids'] = tf.constant([encoded_inputs[
            'token_type_ids']])
        if 'attention_mask' in encoded_inputs:
            encoded_inputs['attention_mask'] = tf.constant([encoded_inputs[
                'attention_mask']])
    elif return_tensors == 'pt' and is_torch_available():
        encoded_inputs['input_ids'] = torch.tensor([encoded_inputs[
            'input_ids']])
        encoded_inputs['token_type_ids'] = torch.tensor([encoded_inputs[
            'token_type_ids']])
        if 'attention_mask' in encoded_inputs:
            encoded_inputs['attention_mask'] = torch.tensor([encoded_inputs
                ['attention_mask']])
    elif return_tensors is not None:
        logger.warning(
            'Unable to convert output to tensors format {}, PyTorch or TensorFlow is not available.'
            .format(return_tensors))
    return encoded_inputs
