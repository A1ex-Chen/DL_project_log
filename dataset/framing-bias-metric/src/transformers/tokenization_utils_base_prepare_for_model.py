@add_end_docstrings(ENCODE_KWARGS_DOCSTRING,
    ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
def prepare_for_model(self, ids: List[int], pair_ids: Optional[List[int]]=
    None, add_special_tokens: bool=True, padding: Union[bool, str,
    PaddingStrategy]=False, truncation: Union[bool, str, TruncationStrategy
    ]=False, max_length: Optional[int]=None, stride: int=0,
    pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[Union[
    str, TensorType]]=None, return_token_type_ids: Optional[bool]=None,
    return_attention_mask: Optional[bool]=None, return_overflowing_tokens:
    bool=False, return_special_tokens_mask: bool=False,
    return_offsets_mapping: bool=False, return_length: bool=False, verbose:
    bool=True, prepend_batch_axis: bool=False, **kwargs) ->BatchEncoding:
    """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens

        Args:
            ids (:obj:`List[int]`):
                Tokenized input ids of the first sequence. Can be obtained from a string by chaining the ``tokenize``
                and ``convert_tokens_to_ids`` methods.
            pair_ids (:obj:`List[int]`, `optional`):
                Tokenized input ids of the second sequence. Can be obtained from a string by chaining the ``tokenize``
                and ``convert_tokens_to_ids`` methods.
        """
    padding_strategy, truncation_strategy, max_length, kwargs = (self.
        _get_padding_truncation_strategies(padding=padding, truncation=
        truncation, max_length=max_length, pad_to_multiple_of=
        pad_to_multiple_of, verbose=verbose, **kwargs))
    pair = bool(pair_ids is not None)
    len_ids = len(ids)
    len_pair_ids = len(pair_ids) if pair else 0
    if return_token_type_ids is not None and not add_special_tokens:
        raise ValueError(
            'Asking to return token_type_ids while setting add_special_tokens to False results in an undefined behavior. Please set add_special_tokens to True or set return_token_type_ids to None.'
            )
    if return_token_type_ids is None:
        return_token_type_ids = 'token_type_ids' in self.model_input_names
    if return_attention_mask is None:
        return_attention_mask = 'attention_mask' in self.model_input_names
    encoded_inputs = {}
    total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(
        pair=pair) if add_special_tokens else 0)
    overflowing_tokens = []
    if (truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and
        max_length and total_len > max_length):
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
        token_type_ids = [0] * len(ids) + ([0] * len(pair_ids) if pair else [])
    encoded_inputs['input_ids'] = sequence
    if return_token_type_ids:
        encoded_inputs['token_type_ids'] = token_type_ids
    if return_special_tokens_mask:
        if add_special_tokens:
            encoded_inputs['special_tokens_mask'
                ] = self.get_special_tokens_mask(ids, pair_ids)
        else:
            encoded_inputs['special_tokens_mask'] = [0] * len(sequence)
    if max_length is None and len(encoded_inputs['input_ids']
        ) > self.model_max_length and verbose:
        if not self.deprecation_warnings.get(
            'sequence-length-is-longer-than-the-specified-maximum', False):
            logger.warning(
                'Token indices sequence length is longer than the specified maximum sequence length for this model ({} > {}). Running this sequence through the model will result in indexing errors'
                .format(len(encoded_inputs['input_ids']), self.
                model_max_length))
        self.deprecation_warnings[
            'sequence-length-is-longer-than-the-specified-maximum'] = True
    if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
        encoded_inputs = self.pad(encoded_inputs, max_length=max_length,
            padding=padding_strategy.value, pad_to_multiple_of=
            pad_to_multiple_of, return_attention_mask=return_attention_mask)
    if return_length:
        encoded_inputs['length'] = len(encoded_inputs['input_ids'])
    batch_outputs = BatchEncoding(encoded_inputs, tensor_type=
        return_tensors, prepend_batch_axis=prepend_batch_axis)
    return batch_outputs
