@add_end_docstrings(ENCODE_KWARGS_DOCSTRING,
    ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
def batch_encode_plus(self, batch_text_or_text_pairs: Union[List[TextInput],
    List[TextInputPair], List[PreTokenizedInput], List[
    PreTokenizedInputPair], List[EncodedInput], List[EncodedInputPair]],
    add_special_tokens: bool=True, padding: Union[bool, str,
    PaddingStrategy]=False, truncation: Union[bool, str, TruncationStrategy
    ]=False, max_length: Optional[int]=None, stride: int=0,
    is_split_into_words: bool=False, pad_to_multiple_of: Optional[int]=None,
    return_tensors: Optional[Union[str, TensorType]]=None,
    return_token_type_ids: Optional[bool]=None, return_attention_mask:
    Optional[bool]=None, return_overflowing_tokens: bool=False,
    return_special_tokens_mask: bool=False, return_offsets_mapping: bool=
    False, return_length: bool=False, verbose: bool=True, **kwargs
    ) ->BatchEncoding:
    """
        Tokenize and prepare for the model a list of sequences or a list of pairs of sequences.

        .. warning::
            This method is deprecated, ``__call__`` should be used instead.

        Args:
            batch_text_or_text_pairs (:obj:`List[str]`, :obj:`List[Tuple[str, str]]`, :obj:`List[List[str]]`, :obj:`List[Tuple[List[str], List[str]]]`, and for not-fast tokenizers, also :obj:`List[List[int]]`, :obj:`List[Tuple[List[int], List[int]]]`):
                Batch of sequences or pair of sequences to be encoded. This can be a list of
                string/string-sequences/int-sequences or a list of pair of string/string-sequences/int-sequence (see
                details in ``encode_plus``).
        """
    padding_strategy, truncation_strategy, max_length, kwargs = (self.
        _get_padding_truncation_strategies(padding=padding, truncation=
        truncation, max_length=max_length, pad_to_multiple_of=
        pad_to_multiple_of, verbose=verbose, **kwargs))
    return self._batch_encode_plus(batch_text_or_text_pairs=
        batch_text_or_text_pairs, add_special_tokens=add_special_tokens,
        padding_strategy=padding_strategy, truncation_strategy=
        truncation_strategy, max_length=max_length, stride=stride,
        is_split_into_words=is_split_into_words, pad_to_multiple_of=
        pad_to_multiple_of, return_tensors=return_tensors,
        return_token_type_ids=return_token_type_ids, return_attention_mask=
        return_attention_mask, return_overflowing_tokens=
        return_overflowing_tokens, return_special_tokens_mask=
        return_special_tokens_mask, return_offsets_mapping=
        return_offsets_mapping, return_length=return_length, verbose=
        verbose, **kwargs)
