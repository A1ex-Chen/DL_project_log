@add_end_docstrings(ENCODE_KWARGS_DOCSTRING,
    ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
def encode_plus(self, text: Union[TextInput, PreTokenizedInput,
    EncodedInput], text_pair: Optional[Union[TextInput, PreTokenizedInput,
    EncodedInput]]=None, add_special_tokens: bool=True, padding: Union[bool,
    str, PaddingStrategy]=False, truncation: Union[bool, str,
    TruncationStrategy]=False, max_length: Optional[int]=None, stride: int=
    0, is_split_into_words: bool=False, pad_to_multiple_of: Optional[int]=
    None, return_tensors: Optional[Union[str, TensorType]]=None,
    return_token_type_ids: Optional[bool]=None, return_attention_mask:
    Optional[bool]=None, return_overflowing_tokens: bool=False,
    return_special_tokens_mask: bool=False, return_offsets_mapping: bool=
    False, return_length: bool=False, verbose: bool=True, **kwargs
    ) ->BatchEncoding:
    """
        Tokenize and prepare for the model a sequence or a pair of sequences.

        .. warning::
            This method is deprecated, ``__call__`` should be used instead.

        Args:
            text (:obj:`str`, :obj:`List[str]` or :obj:`List[int]` (the latter only for not-fast tokenizers)):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                ``tokenize`` method) or a list of integers (tokenized string ids using the ``convert_tokens_to_ids``
                method).
            text_pair (:obj:`str`, :obj:`List[str]` or :obj:`List[int]`, `optional`):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the ``tokenize`` method) or a list of integers (tokenized string ids using the
                ``convert_tokens_to_ids`` method).
        """
    padding_strategy, truncation_strategy, max_length, kwargs = (self.
        _get_padding_truncation_strategies(padding=padding, truncation=
        truncation, max_length=max_length, pad_to_multiple_of=
        pad_to_multiple_of, verbose=verbose, **kwargs))
    return self._encode_plus(text=text, text_pair=text_pair,
        add_special_tokens=add_special_tokens, padding_strategy=
        padding_strategy, truncation_strategy=truncation_strategy,
        max_length=max_length, stride=stride, is_split_into_words=
        is_split_into_words, pad_to_multiple_of=pad_to_multiple_of,
        return_tensors=return_tensors, return_token_type_ids=
        return_token_type_ids, return_attention_mask=return_attention_mask,
        return_overflowing_tokens=return_overflowing_tokens,
        return_special_tokens_mask=return_special_tokens_mask,
        return_offsets_mapping=return_offsets_mapping, return_length=
        return_length, verbose=verbose, **kwargs)
