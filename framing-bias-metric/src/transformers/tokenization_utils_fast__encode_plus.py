def _encode_plus(self, text: Union[TextInput, PreTokenizedInput], text_pair:
    Optional[Union[TextInput, PreTokenizedInput]]=None, add_special_tokens:
    bool=True, padding_strategy: PaddingStrategy=PaddingStrategy.DO_NOT_PAD,
    truncation_strategy: TruncationStrategy=TruncationStrategy.
    DO_NOT_TRUNCATE, max_length: Optional[int]=None, stride: int=0,
    is_split_into_words: bool=False, pad_to_multiple_of: Optional[int]=None,
    return_tensors: Optional[bool]=None, return_token_type_ids: Optional[
    bool]=None, return_attention_mask: Optional[bool]=None,
    return_overflowing_tokens: bool=False, return_special_tokens_mask: bool
    =False, return_offsets_mapping: bool=False, return_length: bool=False,
    verbose: bool=True, **kwargs) ->BatchEncoding:
    batched_input = [(text, text_pair)] if text_pair else [text]
    batched_output = self._batch_encode_plus(batched_input,
        is_split_into_words=is_split_into_words, add_special_tokens=
        add_special_tokens, padding_strategy=padding_strategy,
        truncation_strategy=truncation_strategy, max_length=max_length,
        stride=stride, pad_to_multiple_of=pad_to_multiple_of,
        return_tensors=return_tensors, return_token_type_ids=
        return_token_type_ids, return_attention_mask=return_attention_mask,
        return_overflowing_tokens=return_overflowing_tokens,
        return_special_tokens_mask=return_special_tokens_mask,
        return_offsets_mapping=return_offsets_mapping, return_length=
        return_length, verbose=verbose, **kwargs)
    if return_tensors is None and not return_overflowing_tokens:
        batched_output = BatchEncoding({key: (value[0] if len(value) > 0 and
            isinstance(value[0], list) else value) for key, value in
            batched_output.items()}, batched_output.encodings)
    return batched_output
