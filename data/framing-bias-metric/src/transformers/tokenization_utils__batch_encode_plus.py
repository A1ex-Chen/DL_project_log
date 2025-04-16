def _batch_encode_plus(self, batch_text_or_text_pairs: Union[List[TextInput
    ], List[TextInputPair], List[PreTokenizedInput], List[
    PreTokenizedInputPair], List[EncodedInput], List[EncodedInputPair]],
    add_special_tokens: bool=True, padding_strategy: PaddingStrategy=
    PaddingStrategy.DO_NOT_PAD, truncation_strategy: TruncationStrategy=
    TruncationStrategy.DO_NOT_TRUNCATE, max_length: Optional[int]=None,
    stride: int=0, is_split_into_words: bool=False, pad_to_multiple_of:
    Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=
    None, return_token_type_ids: Optional[bool]=None, return_attention_mask:
    Optional[bool]=None, return_overflowing_tokens: bool=False,
    return_special_tokens_mask: bool=False, return_offsets_mapping: bool=
    False, return_length: bool=False, verbose: bool=True, **kwargs
    ) ->BatchEncoding:

    def get_input_ids(text):
        if isinstance(text, str):
            tokens = self.tokenize(text, **kwargs)
            return self.convert_tokens_to_ids(tokens)
        elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(
            text[0], str):
            if is_split_into_words:
                tokens = list(itertools.chain(*(self.tokenize(t,
                    is_split_into_words=True, **kwargs) for t in text)))
                return self.convert_tokens_to_ids(tokens)
            else:
                return self.convert_tokens_to_ids(text)
        elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(
            text[0], int):
            return text
        else:
            raise ValueError(
                'Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers.'
                )
    if return_offsets_mapping:
        raise NotImplementedError(
            'return_offset_mapping is not available when using Python tokenizers.To use this feature, change your tokenizer to one deriving from transformers.PreTrainedTokenizerFast.'
            )
    input_ids = []
    for ids_or_pair_ids in batch_text_or_text_pairs:
        if not isinstance(ids_or_pair_ids, (list, tuple)):
            ids, pair_ids = ids_or_pair_ids, None
        elif is_split_into_words and not isinstance(ids_or_pair_ids[0], (
            list, tuple)):
            ids, pair_ids = ids_or_pair_ids, None
        else:
            ids, pair_ids = ids_or_pair_ids
        first_ids = get_input_ids(ids)
        second_ids = get_input_ids(pair_ids) if pair_ids is not None else None
        input_ids.append((first_ids, second_ids))
    batch_outputs = self._batch_prepare_for_model(input_ids,
        add_special_tokens=add_special_tokens, padding_strategy=
        padding_strategy, truncation_strategy=truncation_strategy,
        max_length=max_length, stride=stride, pad_to_multiple_of=
        pad_to_multiple_of, return_attention_mask=return_attention_mask,
        return_token_type_ids=return_token_type_ids,
        return_overflowing_tokens=return_overflowing_tokens,
        return_special_tokens_mask=return_special_tokens_mask,
        return_length=return_length, return_tensors=return_tensors, verbose
        =verbose)
    return BatchEncoding(batch_outputs)
