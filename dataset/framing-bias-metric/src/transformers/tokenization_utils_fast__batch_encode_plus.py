def _batch_encode_plus(self, batch_text_or_text_pairs: Union[List[TextInput
    ], List[TextInputPair], List[PreTokenizedInput], List[
    PreTokenizedInputPair]], add_special_tokens: bool=True,
    padding_strategy: PaddingStrategy=PaddingStrategy.DO_NOT_PAD,
    truncation_strategy: TruncationStrategy=TruncationStrategy.
    DO_NOT_TRUNCATE, max_length: Optional[int]=None, stride: int=0,
    is_split_into_words: bool=False, pad_to_multiple_of: Optional[int]=None,
    return_tensors: Optional[str]=None, return_token_type_ids: Optional[
    bool]=None, return_attention_mask: Optional[bool]=None,
    return_overflowing_tokens: bool=False, return_special_tokens_mask: bool
    =False, return_offsets_mapping: bool=False, return_length: bool=False,
    verbose: bool=True) ->BatchEncoding:
    if not isinstance(batch_text_or_text_pairs, list):
        raise TypeError('batch_text_or_text_pairs has to be a list (got {})'
            .format(type(batch_text_or_text_pairs)))
    self.set_truncation_and_padding(padding_strategy=padding_strategy,
        truncation_strategy=truncation_strategy, max_length=max_length,
        stride=stride, pad_to_multiple_of=pad_to_multiple_of)
    encodings = self._tokenizer.encode_batch(batch_text_or_text_pairs,
        add_special_tokens=add_special_tokens, is_pretokenized=
        is_split_into_words)
    tokens_and_encodings = [self._convert_encoding(encoding=encoding,
        return_token_type_ids=return_token_type_ids, return_attention_mask=
        return_attention_mask, return_overflowing_tokens=
        return_overflowing_tokens, return_special_tokens_mask=
        return_special_tokens_mask, return_offsets_mapping=
        return_offsets_mapping, return_length=return_length, verbose=
        verbose) for encoding in encodings]
    sanitized_tokens = {}
    for key in tokens_and_encodings[0][0].keys():
        stack = [e for item, _ in tokens_and_encodings for e in item[key]]
        sanitized_tokens[key] = stack
    sanitized_encodings = [e for _, item in tokens_and_encodings for e in item]
    if return_overflowing_tokens:
        overflow_to_sample_mapping = []
        for i, (toks, _) in enumerate(tokens_and_encodings):
            overflow_to_sample_mapping += [i] * len(toks['input_ids'])
        sanitized_tokens['overflow_to_sample_mapping'
            ] = overflow_to_sample_mapping
    return BatchEncoding(sanitized_tokens, sanitized_encodings, tensor_type
        =return_tensors)
